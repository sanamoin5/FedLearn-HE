import copy
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
from pympler import asizeof
from torchvision.datasets.mnist import MNIST

from .FedClient import FedClient, measure_time


class BatchCryptBasedFedAvgClient(FedClient):
    def __init__(
        self,
        args: Namespace,
        train_dataset: MNIST,
        user_groups: Dict[int, Set[int64]],
        json_logs,
        server: Type[FedAvgServer],
    ) -> None:
        super().__init__(args, train_dataset, user_groups, json_logs, server)
        self.args = args
        self.print_every = 2

    @measure_time
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

    @measure_time
    def quantize_weights(
        self, clients_weights: List[OrderedDict]
    ) -> Tuple[List[ndarray], List[float64]]:
        for idx in range(len(clients_weights)):
            clients_weights[idx] = [
                clients_weights[idx][x].cpu().numpy() for x in clients_weights[idx]
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

    def change_back_to_ordered_dict(self, clients_weights, layer_names):
        client_weights_ordereddict = []
        for client_weights in clients_weights:
            ordered_dict = OrderedDict()
            for i, layer_name in enumerate(layer_names):
                ordered_dict[layer_name] = torch.from_numpy(np.asarray(client_weights[i]))
            client_weights_ordereddict.append(ordered_dict)
        return client_weights_ordereddict

    @measure_time
    def aggregate_by_weights(
        self, local_weights: List[OrderedDict]
    ) -> Tuple[OrderedDict, Dict[str, float]]:
        local_weights = copy.deepcopy(local_weights)
        self.json_logs["local_weights_size"] = asizeof.asizeof(local_weights)

        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}
        quantize_weights, r_maxs = self.quantize_weights(local_weights)
        quantize_weights = self.change_back_to_ordered_dict(
            quantize_weights, weight_shapes.keys()
        )
        self.json_logs["quantize_weights_size"] = asizeof.asizeof(quantize_weights)
        # Encrypt client weights
        encrypted_weights = self.encrypt_weights(quantize_weights)
        self.json_logs["encrypted_weights_size"] = asizeof.asizeof(encrypted_weights)

        # Aggregate encrypted weights
        global_averaged_encrypted_weights = self.aggregate_weights(encrypted_weights)
        self.json_logs["global_averaged_encrypted_weights_size"] = asizeof.asizeof(
            global_averaged_encrypted_weights
        )
        # Decrypt and average the weights

        global_averaged_weights = self.decrypt_weights(
            global_averaged_encrypted_weights, weight_shapes
        )
        self.json_logs["global_averaged_decr_weights"] = asizeof.asizeof(
            global_averaged_weights
        )

        dequantize_weights = self.dequantize_weights(global_averaged_weights, r_maxs)
        self.json_logs["dequantize_weights_weights"] = asizeof.asizeof(
            dequantize_weights
        )

        ordered_dict = OrderedDict()
        for i, layer_name in enumerate(weight_shapes.keys()):
            ordered_dict[layer_name] = torch.from_numpy(
                np.asarray(dequantize_weights[i])
            )

        return ordered_dict
