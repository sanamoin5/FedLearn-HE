import copy
import time
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List, Set, Tuple, Type

import numpy as np
import torch
import torch.utils.data.dataloader
from models.quant_utils import (
    calculate_clip_threshold_aciq_g,
    clip_with_threshold,
    quantize_per_layer,
    unquantize_matrix,
)
from models.server import FedAvgServer
from numpy import float64, int64, ndarray
from tensorboardX.writer import SummaryWriter
from torchvision.datasets.mnist import MNIST

from .FedClient import FedClient


class BatchCryptBasedFedAvgClient(FedClient):
    def __init__(
        self,
        args: Namespace,
        train_dataset: MNIST,
        user_groups: Dict[int, Set[int64]],
        logger: SummaryWriter,
        server: Type[FedAvgServer],
    ) -> None:
        super().__init__(args, train_dataset, user_groups, logger, server)
        self.args = args
        self.print_every = 2

        # Initialize train, validation and test loaders for each client
        self.train_loader, self.valid_loader, self.test_loader = [], [], []
        for idx in range(self.args.num_clients):
            train_loader, valid_loader, test_loader = self.train_val_test(
                list(self.user_groups[idx])
            )
            self.train_loader.append(train_loader)
            self.valid_loader.append(valid_loader)
            self.test_loader.append(test_loader)

    def quantize_weights(
        self, clients_weights: List[OrderedDict]
    ) -> Tuple[List[ndarray], List[float64]]:
        for idx in range(len(clients_weights)):
            clients_weights[idx] = [
                clients_weights[idx][x].numpy() for x in clients_weights[idx]
            ]

        sizes = [
            np.prod(item.shape) * self.args.num_clients for item in clients_weights[0]
        ]
        max_values = []
        min_values = []
        for layer_idx in range(len(clients_weights[0])):
            max_values.append([np.max([item[layer_idx] for item in clients_weights])])
            min_values.append([np.min([item[layer_idx] for item in clients_weights])])
        grads_max_min = np.concatenate(
            [np.array(max_values), np.array(min_values)], axis=1
        )
        q_width = 16
        clipping_thresholds = calculate_clip_threshold_aciq_g(
            grads_max_min, sizes, bit_width=q_width
        )

        r_maxs = [x * self.args.num_clients for x in clipping_thresholds]

        clients_weights = [
            clip_with_threshold(item, clipping_thresholds) for item in clients_weights
        ]
        clients_weights = [
            quantize_per_layer(item, r_maxs, bit_width=q_width)
            for item in clients_weights
        ]

        return clients_weights, r_maxs

    def dequantize_weights(
        self, grads: Dict[str, torch.Tensor], r_maxs: List[float64], q_width: int = 16
    ) -> ndarray:
        grads = [grads[x].numpy() for x in grads]
        result = []
        for component, r_max in zip(grads, r_maxs):
            result.append(
                unquantize_matrix(component, bit_width=q_width, r_max=r_max).astype(
                    np.float32
                )
            )
        return np.array(result)

    def change_back_to_ordered_dict(self, clients_weights, layer_names):
        client_weights_ordereddict = []
        for client_weights in clients_weights:
            ordered_dict = OrderedDict()
            for i, layer_name in enumerate(layer_names):
                ordered_dict[layer_name] = client_weights[i]
            client_weights_ordereddict.append(ordered_dict)
        return client_weights_ordereddict

    def aggregate_by_weights(
        self, local_weights: List[OrderedDict]
    ) -> Tuple[OrderedDict, Dict[str, float]]:
        local_weights = copy.deepcopy(local_weights)
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}
        time_quant_start = time.time()
        quantize_weights, r_maxs = self.quantize_weights(local_weights)
        quantize_weights = self.change_back_to_ordered_dict(
            quantize_weights, weight_shapes.keys()
        )
        time_quant_end = time.time()
        # Encrypt client weights
        time_encr_start = time.time()
        encrypted_weights = self.he.encrypt_client_weights(quantize_weights)
        time_encr_end = time.time()
        # Aggregate encrypted weights
        time_avg_start = time.time()
        global_averaged_encrypted_weights = self.server(encrypted_weights).aggregate(
            self.args.he_scheme_name
        )
        time_avg_end = time.time()
        # Decrypt and average the weights
        time_decr_start = time.time()
        global_averaged_weights = self.he.decrypt_and_average_weights(
            global_averaged_encrypted_weights,
            weight_shapes,
            client_weights=self.args.num_clients,
            secret_key=self.secret_key,
        )
        time_decr_end = time.time()
        time_dequant_start = time.time()
        dequantize_weights = self.dequantize_weights(global_averaged_weights, r_maxs)
        ordered_dict = OrderedDict()
        for i, layer_name in enumerate(weight_shapes.keys()):
            ordered_dict[layer_name] = torch.from_numpy(dequantize_weights[i])
        time_dequant_end = time.time()

        return ordered_dict, {
            "time_quant": time_quant_end - time_quant_start,
            "time_encr": time_encr_end - time_encr_start,
            "time_avg": time_avg_end - time_avg_start,
            "time_decr": time_decr_end - time_decr_start,
            "time_dequant": time_dequant_end - time_dequant_start,
        }
