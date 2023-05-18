import copy
import math
import time
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
import torch.utils.data.dataloader
from homomorphic_encryption import HEScheme
from nn_models import MnistModel
from numpy import float64, int64, ndarray
from scipy import optimize as opt
from tensorboardX.writer import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.mnist import MNIST


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset: MNIST, idxs: List[int64]) -> None:
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class FedClient:
    def __init__(
        self,
        args: Namespace,
        dataset: MNIST,
        user_groups: Dict[int, Set[int64]],
        logger: SummaryWriter,
        server: Type[FedAvgServer],
    ) -> None:
        self.args = args
        self.device = "cuda" if args.gpu else "cpu"
        self.dataset = dataset
        self.user_groups = user_groups
        self.logger = logger
        self.server = server
        self.criterion = nn.NLLLoss().to(self.device)

        # initialize homomorphic encryption
        self.he = HEScheme(self.args.he_scheme_name)
        if self.args.he_scheme_name == "ckks":
            self.context = self.he.context
            self.secret_key = (
                self.context.secret_key()
            )  # save the secret key before making context public
            self.context.make_context_public()  # make the context object public so it can be shared across clients
        elif self.args.he_scheme_name == "paillier":
            self.secret_key = self.he.private_key
        elif self.args.he_scheme_name == "bfv":
            self.context = self.he.context
            self.secret_key = (
                self.context.secret_key()
            )  # save the secret key before making context public
            self.context.make_context_public()

    def train_val_test(
        self, idxs: List[int64]
    ) -> Tuple[
        torch.utils.data.dataloader.DataLoader,
        torch.utils.data.dataloader.DataLoader,
        torch.utils.data.dataloader.DataLoader,
    ]:
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[: int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)) :]

        trainloader = DataLoader(
            DatasetSplit(self.dataset, idxs_train),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        validloader = DataLoader(
            DatasetSplit(self.dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
        )
        testloader = DataLoader(
            DatasetSplit(self.dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
        )
        return trainloader, validloader, testloader

    def evaluate_model(
        self, model: MnistModel, testloader: torch.utils.data.dataloader.DataLoader
    ) -> Tuple[float, float]:
        """Returns the inference accuracy and loss."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

    def get_client_weights(self):
        if self.args.client_weights:
            return self.args.client_weights
        else:
            return [len(value) for user, value in self.user_groups.items()]


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

    def train_local_model(self, model, global_round, client_idx):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        # Iterate over each epoch
        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_loader[client_idx]):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the gradients
                model.zero_grad()

                # Compute the forward pass
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # Print the loss and global round every `print_every` batches
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            epoch,
                            batch_idx * len(images),
                            len(self.train_loader[client_idx].dataset),
                            100.0 * batch_idx / len(self.train_loader[client_idx]),
                            loss.item(),
                        )
                    )

                # Log the loss to TensorBoard
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model, sum(epoch_loss) / len(epoch_loss)

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        # Aggregate local model weights into global model weights
        aggregated_weights, time_processing = self.aggregate_by_weights(local_weights)

        if self.args.train_without_encryption:
            global_weights = self.server.average_weights_nonencrypted(local_weights)
            global_model_non_encry.load_state_dict(global_weights)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)

        return (
            train_loss,
            train_accuracy,
            aggregated_weights,
            global_model_non_encry,
            time_processing,
        )


class WeightedFedAvgClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, logger, server):
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

    def aggregate_by_weights(self, local_weights, client_weights):
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}

        # Encrypt client weights
        encrypted_weights = self.he.encrypt_client_weights(local_weights)

        # Aggregate encrypted weights
        global_averaged_encrypted_weights = self.server(
            encrypted_weights, client_weights
        ).aggregate(self.args.he_scheme_name)

        # Decrypt and average the weights
        global_averaged_weights = self.he.decrypt_and_average_weights(
            global_averaged_encrypted_weights,
            weight_shapes,
            client_weights=sum(client_weights),
            secret_key=self.secret_key,
        )
        return global_averaged_weights

    def train_local_model(self, model, global_round, client_idx):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        # Iterate over each epoch
        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_loader[client_idx]):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the gradients
                model.zero_grad()

                # Compute the forward pass
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # Print the loss and global round every `print_every` batches
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            epoch,
                            batch_idx * len(images),
                            len(self.train_loader[client_idx].dataset),
                            100.0 * batch_idx / len(self.train_loader[client_idx]),
                            loss.item(),
                        )
                    )

                # Log the loss to TensorBoard
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model, sum(epoch_loss) / len(epoch_loss)

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        client_weights = self.get_client_weights()
        # Aggregate local model weights into global model weights
        aggregated_weights = self.aggregate_by_weights(local_weights, client_weights)

        if self.args.train_without_encryption:
            global_weights = self.server.average_weights_nonencrypted(local_weights)
            global_model_non_encry.load_state_dict(global_weights)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)

        return train_loss, train_accuracy, aggregated_weights, global_model_non_encry


class GradientBasedFedAvgClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, logger, server):
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

    def aggregate_by_gradients(self, local_gradients, client_weights):
        # Get shapes of gradients
        weight_shapes = {k: v.shape for k, v in local_gradients[0].items()}

        # Encrypt client gradients
        encrypted_gradients = self.he.encrypt_client_weights(local_gradients)

        # Aggregate encrypted gradients
        global_averaged_encrypted_gradients = self.server(
            encrypted_gradients, client_weights
        ).aggregate(self.args.he_scheme_name)

        # Decrypt and average the gradients
        global_decrypted_averaged_gradients = self.he.decrypt_and_average_weights(
            global_averaged_encrypted_gradients,
            weight_shapes,
            client_weights=sum(client_weights),
            secret_key=self.secret_key,
        )
        return global_decrypted_averaged_gradients

    def train_local_model(self, model, global_round, client_idx):
        # Set mode to train model
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        # Iterate over each epoch
        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_loader[client_idx]):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the gradients
                model.zero_grad()

                # Compute the forward pass
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()

                # Print the loss and global round every `print_every` batches
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            epoch,
                            batch_idx * len(images),
                            len(self.train_loader[client_idx].dataset),
                            100.0 * batch_idx / len(self.train_loader[client_idx]),
                            loss.item(),
                        )
                    )

                # Log the loss to TensorBoard
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model, sum(epoch_loss) / len(epoch_loss)

    def get_grads_for_local_models(self, local_weights, global_model):
        """return a dict: {name: params}"""
        local_grads = []
        for local_weights_cli in local_weights:
            state_dict = {}
            for name, param0 in global_model.state_dict().items():
                param1 = local_weights_cli[name]
                state_dict[name] = param0.detach() - param1.detach()
            local_grads.append(state_dict)
        return local_grads

    def update_model_weights_with_aggregated_grads(self, aggregated_grads, model):
        weights = model.state_dict()

        for name, grad in aggregated_grads.items():
            weights[name] -= grad

        return weights

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        local_grads = self.get_grads_for_local_models(local_weights, global_model)
        client_weights = self.get_client_weights()
        # Aggregate local model weights into global model weights
        aggregated_grads = self.aggregate_by_gradients(local_grads, client_weights)

        global_model_weights = self.update_model_weights_with_aggregated_grads(
            aggregated_grads, global_model
        )
        if self.args.train_without_encryption:
            aggregated_grads_nonencr = self.server.average_weights_nonencrypted(
                local_grads
            )
            global_model_weights_nonencr = (
                self.update_model_weights_with_aggregated_grads(
                    aggregated_grads_nonencr, global_model_non_encry
                )
            )
            global_model_non_encry.load_state_dict(global_model_weights_nonencr)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)
        return train_loss, train_accuracy, global_model_weights, global_model_non_encry


class QuantFedClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, logger, server):
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

    @staticmethod
    def quantize_matrix_stochastic(matrix, bit_width=8, r_max=0.5):
        og_sign = np.sign(matrix)
        uns_matrix = np.multiply(matrix, og_sign)
        uns_result = np.multiply(
            uns_matrix, np.divide((pow(2, bit_width - 1) - 1.0), r_max)
        )
        result = np.multiply(og_sign, uns_result)
        return result, og_sign

    @staticmethod
    def unquantize_matrix(matrix, bit_width=8, r_max=0.5):
        matrix = matrix.astype(int)
        og_sign = np.sign(matrix)
        uns_matrix = np.multiply(matrix, og_sign)
        uns_result = np.multiply(
            uns_matrix, np.divide(r_max, (pow(2, bit_width - 1) - 1.0))
        )
        result = og_sign * uns_result
        return result.astype(np.float32)

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

        print("clipping_thresholds", clipping_thresholds)

        r_maxs = [x * self.args.num_clients for x in clipping_thresholds]

        def clip_with_threshold(grads, thresholds):
            return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]

        clients_weights = [
            clip_with_threshold(item, clipping_thresholds) for item in clients_weights
        ]

        def quantize_per_layer(party, r_maxs, bit_width=16):
            result = []
            for component, r_max in zip(party, r_maxs):
                x, _ = self.quantize_matrix_stochastic(
                    component, bit_width=bit_width, r_max=r_max
                )
                result.append(x)
            return np.array(result)

        q_width = 16
        clients_weights = [
            quantize_per_layer(item, r_maxs, bit_width=q_width)
            for item in clients_weights
        ]

        return clients_weights, r_maxs

    def dequantize_weights(self, grads, r_maxs, q_width=16):
        grads = [grads[x].numpy() for x in grads]
        result = []
        for component, r_max in zip(grads, r_maxs):
            result.append(
                self.unquantize_matrix(
                    component, bit_width=q_width, r_max=r_max
                ).astype(np.float32)
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

    def train_local_model(self, model, global_round, client_idx):
        start_time = time.time()
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        # Iterate over each epoch
        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_loader[client_idx]):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the gradients
                model.zero_grad()

                # Compute the forward pass
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # Print the loss and global round every `print_every` batches
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            epoch,
                            batch_idx * len(images),
                            len(self.train_loader[client_idx].dataset),
                            100.0 * batch_idx / len(self.train_loader[client_idx]),
                            loss.item(),
                        )
                    )

                # Log the loss to TensorBoard
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        end_time = time.time()
        return model, sum(epoch_loss) / len(epoch_loss), end_time - start_time

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss, time_training = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        # Aggregate local model weights into global model weights
        aggregated_weights, time_processing = self.aggregate_by_weights(local_weights)
        time_processing["time_training_local"] = time_training
        if self.args.train_without_encryption:
            global_weights = self.server.average_weights_nonencrypted(local_weights)
            global_model_non_encry.load_state_dict(global_weights)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)

        return (
            train_loss,
            train_accuracy,
            aggregated_weights,
            global_model_non_encry,
            time_processing,
        )


def mse_laplace(alpha: Union[float, float64], b: float, num_bits: int) -> float64:
    """
    Calculating the sum of clipping error and quantization error for Laplace case

    Args:
    alpha: the clipping value
    b: location parameter of Laplace distribution
    num_bits: number of bits used for quantization

    Return:
    The sum of clipping error and quantization error
    """
    return 2 * (b**2) * np.exp(-alpha / b) + (
        2 * alpha**2 * (2**num_bits - 2)
    ) / (3 * (2 ** (3 * num_bits)))


def mse_gaussian(alpha: Union[float, float64], sigma: float, num_bits: int) -> float64:
    """
    Calculating the sum of clipping error and quantization error for Gaussian case

    Args:
    alpha: the clipping value
    sigma: scale parameter parameter of Gaussian distribution
    num_bits: number of bits used for quantization

    Return:
    The sum of clipping error and quantization error
    """
    clipping_err = (sigma**2 + (alpha**2)) * (
        1 - math.erf(alpha / (sigma * np.sqrt(2.0)))
    ) - np.sqrt(2.0 / np.pi) * alpha * sigma * (
        np.e ** ((-1) * (0.5 * (alpha**2)) / sigma**2)
    )
    quant_err = (2 * alpha**2 * (2**num_bits - 2)) / (3 * (2 ** (3 * num_bits)))
    return clipping_err + quant_err


# To facilitate calculations, we avoid calculating MSEs from scratch each time.
# Rather, as N (0, sigma^2) = sigma * N (0, 1) and Laplace(0, b) = b * Laplace(0, 1),
# it is sufficient to store the optimal clipping values for N (0, 1) and Laplace(0, 1) and scale these
# values by sigma and b, which are estimated from the tensor values.

# Given b = 1, for laplace distribution
b = 1.0
# print("Optimal alpha coeficients for laplace case, while num_bits falls in [2, 8].")
alphas = []
for m in range(2, 33, 1):
    alphas.append(opt.minimize_scalar(lambda x: mse_laplace(x, b=b, num_bits=m)).x)
# print(np.array(alphas))

# Given sigma = 1, for Gaussian distribution
sigma = 1.0
# print("Optimal alpha coeficients for gaussian clipping, while num_bits falls in [2, 8]")
alphas = []
for m in range(2, 33, 1):
    alphas.append(
        opt.minimize_scalar(lambda x: mse_gaussian(x, sigma=sigma, num_bits=m)).x
    )
# print(np.array(alphas))


def get_alpha_laplace(values, num_bits):
    """
    Calculating optimal alpha(clipping value) in Laplace case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    """

    # Dictionary that stores optimal clipping values for Laplace(0, 1)
    alpha_laplace = {
        2: 2.83068299,
        3: 3.5773953,
        4: 4.56561968,
        5: 5.6668432,
        6: 6.83318852,
        7: 8.04075143,
        8: 9.27621011,
        9: 10.53164388,
        10: 11.80208734,
        11: 13.08426947,
        12: 14.37593053,
        13: 15.67544068,
        14: 16.98157905,
        15: 18.29340105,
        16: 19.61015778,
        17: 20.93124164,
        18: 22.25615278,
        19: 23.58447327,
        20: 24.91584992,
        21: 26.24998231,
        22: 27.58661098,
        23: 28.92551169,
        24: 30.26648869,
        25: 31.60937055,
        26: 32.9540057,
        27: 34.30026003,
        28: 35.64801378,
        29: 36.99716035,
        30: 38.3476039,
        31: 39.69925781,
        32: 41.05204406,
    }

    # That's how ACIQ paper calcualte b
    b = np.mean(np.abs(values - np.mean(values)))
    return alpha_laplace[num_bits] * b


def get_alpha_gaus(values: ndarray, values_size: int64, num_bits: int) -> float64:
    """
    Calculating optimal alpha(clipping value) in Gaussian case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    """

    # Dictionary that stores optimal clipping values for N(0, 1)
    alpha_gaus = {
        2: 1.71063516,
        3: 2.02612148,
        4: 2.39851063,
        5: 2.76873681,
        6: 3.12262004,
        7: 3.45733738,
        8: 3.77355322,
        9: 4.07294252,
        10: 4.35732563,
        11: 4.62841243,
        12: 4.88765043,
        13: 5.1363822,
        14: 5.37557768,
        15: 5.60671468,
        16: 5.82964388,
        17: 6.04501354,
        18: 6.25385785,
        19: 6.45657762,
        20: 6.66251328,
        21: 6.86053901,
        22: 7.04555454,
        23: 7.26136857,
        24: 7.32861916,
        25: 7.56127906,
        26: 7.93151212,
        27: 7.79833847,
        28: 7.79833847,
        29: 7.9253003,
        30: 8.37438905,
        31: 8.37438899,
        32: 8.37438896,
    }
    # That's how ACIQ paper calculate sigma, based on the range (efficient but not accurate)
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    # sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / (
        (2 * np.log(values_size)) ** 0.5
    )
    return alpha_gaus[num_bits] * sigma


def calculate_clip_threshold_aciq_g(
    grads: ndarray, grads_sizes: List[int64], bit_width: int = 8
) -> List[float64]:
    print("ACIQ bit width:", bit_width)
    res = []
    for idx in range(len(grads)):
        res.append(get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    # return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
    return res


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

    @staticmethod
    def unquantize_matrix(
        matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
    ) -> ndarray:
        matrix = matrix.astype(int)
        og_sign = np.sign(matrix)
        uns_matrix = np.multiply(matrix, og_sign)
        uns_result = np.multiply(
            uns_matrix, np.divide(r_max, (pow(2, bit_width - 1) - 1.0))
        )
        result = og_sign * uns_result
        return result.astype(np.float32)

    @staticmethod
    def quantize_matrix_stochastic(
        matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
    ) -> Tuple[ndarray, ndarray]:
        og_sign = np.sign(matrix)
        uns_matrix = np.multiply(matrix, og_sign)
        uns_result = np.multiply(
            uns_matrix, np.divide((pow(2, bit_width - 1) - 1.0), r_max)
        )
        result = np.multiply(og_sign, uns_result)
        return result, og_sign

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

        def clip_with_threshold(grads, thresholds):
            return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]

        def quantize_per_layer(party, r_maxs, bit_width=16):
            result = []
            for component, r_max in zip(party, r_maxs):
                x, _ = self.quantize_matrix_stochastic(
                    component, bit_width=bit_width, r_max=r_max
                )
                result.append(x)
            return np.array(result)

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
                self.unquantize_matrix(
                    component, bit_width=q_width, r_max=r_max
                ).astype(np.float32)
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

    def train_local_model(
        self, model: MnistModel, global_round: int, client_idx: int
    ) -> Tuple[MnistModel, float, float]:
        start_time = time.time()
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        # Iterate over each epoch
        for epoch in range(self.args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_loader[client_idx]):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the gradients
                model.zero_grad()

                # Compute the forward pass
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # Print the loss and global round every `print_every` batches
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            client_idx,
                            epoch,
                            batch_idx * len(images),
                            len(self.train_loader[client_idx].dataset),
                            100.0 * batch_idx / len(self.train_loader[client_idx]),
                            loss.item(),
                        )
                    )

                # Log the loss to TensorBoard
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        end_time = time.time()
        return model, sum(epoch_loss) / len(epoch_loss), end_time - start_time

    def train_clients(
        self,
        global_model: MnistModel,
        round_: int,
        global_model_non_encry: Optional[MnistModel] = None,
    ) -> Tuple[float, float, OrderedDict, MnistModel, Dict[str, float]]:
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss, time_training = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        # Aggregate local model weights into global model weights
        aggregated_weights, time_processing = self.aggregate_by_weights(local_weights)
        time_processing["time_training_local"] = time_training
        if self.args.train_without_encryption:
            global_weights = self.server.average_weights_nonencrypted(local_weights)
            global_model_non_encry.load_state_dict(global_weights)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)

        return (
            train_loss,
            train_accuracy,
            aggregated_weights,
            global_model_non_encry,
            time_processing,
        )
