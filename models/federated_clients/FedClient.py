import copy
import time
from abc import abstractmethod
from argparse import Namespace
from typing import Dict, List, Set, Tuple, Type

import torch
import torch.utils.data.dataloader
from models.homomorphic_encryption import HEScheme
from models.nn_models import MnistModel
from models.server import FedAvgServer
from numpy import int64
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

    def train_local_model(
        self, model: MnistModel, global_round: int, client_idx: int
    ) -> Tuple[MnistModel, float, float]:
        start_time = time.time()
        global_model = copy.deepcopy(model)

        # Set mode to train model
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
        end_time = time.time()
        return model, sum(epoch_loss) / len(epoch_loss), end_time - start_time

    @abstractmethod
    def aggregate_by_weights(self, weights):
        pass

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []
        time_training = 0
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
