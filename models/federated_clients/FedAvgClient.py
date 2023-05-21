import time
from argparse import Namespace
from typing import Dict, Set, Type

from models.server import FedAvgServer
from numpy import int64
from tensorboardX.writer import SummaryWriter
from torchvision.datasets.mnist import MNIST

from .FedClient import FedClient


class FedAvgClient(FedClient):
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

    def aggregate_by_weights(self, local_weights):
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}

        # Encrypt client weights
        time_encr_start = time.time()
        encrypted_weights = self.he.encrypt_client_weights(local_weights)
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
        return global_averaged_weights, {
            "time_encr": time_encr_end - time_encr_start,
            "time_avg": time_avg_end - time_avg_start,
            "time_decr": time_decr_end - time_decr_start,
        }
