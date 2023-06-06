from argparse import Namespace
from typing import Dict, Set, Type

from models.server import FedAvgServer
from numpy import int64
from pympler import asizeof
from torchvision.datasets.mnist import MNIST

from .FedClient import FedClient


class FedAvgClient(FedClient):
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

    def aggregate_by_weights(self, local_weights):
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}
        self.json_logs["local_weights_size"] = asizeof.asizeof(local_weights)

        # Encrypt client weights
        encrypted_weights = self.encrypt_weights(local_weights)
        self.json_logs["encrypted_weights_size"] = asizeof.asizeof(encrypted_weights)

        # Aggregate encrypted weights
        global_averaged_encrypted_weights = self.aggregate_weights(encrypted_weights)
        self.json_logs["global_averaged_encrypted_weightssize"] = asizeof.asizeof(
            global_averaged_encrypted_weights
        )

        # Decrypt and average the weights
        global_averaged_weights = self.decrypt_weights(
            global_averaged_encrypted_weights, weight_shapes
        )
        self.json_logs["global_averaged_decr_weights"] = asizeof.asizeof(
            global_averaged_weights
        )

        return global_averaged_weights
