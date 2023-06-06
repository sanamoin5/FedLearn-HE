import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data.dataloader
from models.quant_utils import (
    clip_with_threshold,
    quantize_per_layer,
    unquantize_matrix,
)
from pympler import asizeof

from .FedClient import FedClient, measure_time


class QuantFedClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, json_logs, server):
        super().__init__(args, train_dataset, user_groups, json_logs, server)
        self.args = args
        self.print_every = 2

    @measure_time
    def quantize_weights(self, clients_weights):
        for idx in range(len(clients_weights)):
            clients_weights[idx] = [
                clients_weights[idx][x].numpy() for x in clients_weights[idx]
            ]

        # clipping_thresholds = encryption.calculate_clip_threshold(grads_0)
        theta = 2.5
        clients_weights_mean = []
        clients_weights_mean_square = []
        for client_idx in range(len(clients_weights)):
            temp_mean = [
                np.mean(clients_weights[client_idx][layer_idx])
                for layer_idx in range(len(clients_weights[client_idx]))
            ]
            temp_mean_square = [
                np.mean(clients_weights[client_idx][layer_idx] ** 2)
                for layer_idx in range(len(clients_weights[client_idx]))
            ]
            clients_weights_mean.append(temp_mean)
            clients_weights_mean_square.append(temp_mean_square)
        clients_weights_mean = np.array(clients_weights_mean)
        clients_weights_mean_square = np.array(clients_weights_mean_square)

        layers_size = np.array([_.size for _ in clients_weights[0]])
        clipping_thresholds = (
            theta
            * (
                np.sum(clients_weights_mean_square * layers_size, 0)
                / (layers_size * self.args.num_clients)
                - (
                    np.sum(clients_weights_mean * layers_size, 0)
                    / (layers_size * self.args.num_clients)
                )
                ** 2
            )
            ** 0.5
        )

        r_maxs = [x * self.args.num_clients for x in clipping_thresholds]

        clients_weights = [
            clip_with_threshold(item, clipping_thresholds) for item in clients_weights
        ]

        q_width = 16
        clients_weights = [
            quantize_per_layer(item, r_maxs, bit_width=q_width)
            for item in clients_weights
        ]

        return clients_weights, r_maxs

    @measure_time
    def dequantize_weights(self, grads, r_maxs, q_width=16):
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

    def aggregate_by_weights(self, local_weights):
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
        self.json_logs["global_averaged_decr_weights_size"] = asizeof.asizeof(
            global_averaged_weights
        )

        dequantize_weights = self.dequantize_weights(global_averaged_weights, r_maxs)
        self.json_logs["dequantize_weights_size"] = asizeof.asizeof(dequantize_weights)

        ordered_dict = OrderedDict()
        for i, layer_name in enumerate(weight_shapes.keys()):
            ordered_dict[layer_name] = torch.from_numpy(dequantize_weights[i])

        return ordered_dict
