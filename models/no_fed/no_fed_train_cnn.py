import wandb
from tqdm import tqdm
import json
from args_parser import args_parser
import sys
import os

import numpy as np
from torchvision import datasets, transforms
import copy
import time

import torch
import torch.utils.data.dataloader
from cifar_resnet import CIFAR10_ResNet
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.mnist import MNIST


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset: MNIST, idxs) -> None:
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class ClientTrainer:
    def __init__(self):
        self.args = args_parser()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        wandb.init(project="FedLearn-HE-Experiments")
        wandb.config.update({
            "learning_rate": self.args.lr,
            "epochs": self.args.epochs,
            "batch_size": self.args.local_bs,
            "optimizer": self.args.optimizer,
            "model": self.args.model,
            "num_clients": self.args.num_clients,
            "iid": self.args.iid,
            "device": self.device
        })

        self.train_dataset, self.test_dataset, self.user_groups = self.get_datasets_and_user_groups(
            args.num_clients)
        self.dataset = self.train_dataset
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.test_dataset_bs,
            shuffle=True,
        )

        self.model = CIFAR10_ResNet

    def train_val(self, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[: int(0.9 * len(idxs))]
        idxs_val = idxs[int(0.9 * len(idxs)):]

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

        return trainloader, validloader

    @staticmethod
    def dataset_iid(dataset, num_clients, seed=42):
        """
        Sample I.I.D. client data from CIFAR10 or MNIST or BALNMP dataset
        :param dataset:
        :param num_clients:
        :return: dict of image index
        """
        np.random.seed(seed)
        num_items = int(len(dataset) / num_clients)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    @staticmethod
    def cifar_noniid(dataset, num_clients):
        """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_clients:
        :return:
        """
        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_clients)}
        idxs = np.arange(num_shards * num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = np.array(dataset.train_labels)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(num_clients):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0
                )
        return dict_users

    def get_datasets_and_user_groups(self, num_clients):
        train_dataset, test_dataset, user_groups = None, None, None
        if self.args.model == "cifar_cnn" or self.args.model == "cifar_resnet":
            data_dir = "data/cifar/"
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train
            )

            test_dataset = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test
            )

            # sample training data amongst users
            if self.args.iid:
                # Sample IID user data from Mnist
                user_groups = self.dataset_iid(train_dataset, self.args.num_clients)
            else:
                user_groups = self.cifar_noniid(train_dataset, self.args.num_clients)

        return train_dataset, test_dataset, user_groups

    def train_and_evaluate(self):
        all_train_accuracies, all_valid_accuracies, test_accuracies = [], [], []

        for client_idx in range(self.args.num_clients):
            print(f"Training client {client_idx}")
            client_model = copy.deepcopy(self.model.to(self.device))
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            train_accuracies, valid_accuracies, test_accuracies = self.train(client_model, optimizer, criterion, client_idx)
            all_train_accuracies.append(train_accuracies)
            all_valid_accuracies.append(valid_accuracies)


        return all_train_accuracies, all_valid_accuracies, test_accuracies

    def train(self, model, optimizer, criterion, client_idx):
        model.train()
        metrics = {"epochs": [], "test_metrics": []}

        train_loader, valid_loader = self.train_val(list(self.user_groups[client_idx]))

        train_accuracies, valid_accuracies, test_accuracies = [], [], []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=float(self.args.lr), momentum=float(self.args.momentum),
                weight_decay=float(self.args.weight_decay)
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(self.args.lr), weight_decay=3.2e-6
            )

        # Learning rate scheduler
        scheduler = None
        if self.args.lr_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in range(self.args.epochs):
            epoch_loss=0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}, Client {client_idx}")
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if scheduler is not None:
                    scheduler.step()

                progress_bar.set_postfix(loss=epoch_loss / len(train_loader))
            if (epoch + 1) % 20 == 0:
                print(f'Running test at Epoch: {epoch + 1}')
                test_accuracy = self.evaluate(model, self.test_dataloader)
                print(f'Epoch: {epoch + 1}, Test Accuracy: {test_accuracy}')
                wandb.log({"Client": client_idx, "Epoch": epoch, "Test Accuracy": test_accuracy})
                test_accuracies.append(test_accuracy)

                test_metrics = {
                    "epoch": epoch + 1,
                    "test_accuracy": test_accuracy
                }
                metrics["test_metrics"].append(test_metrics)


            # Evaluate on training and validation set after each epoch
            train_accuracy = self.evaluate(model, train_loader)
            valid_accuracy = self.evaluate(model, valid_loader)
            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)
            print(
                f'Epoch: {epoch}, Client: {client_idx}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}, Val Acc: {valid_accuracy:.2f}')

            wandb.log({"Client": client_idx, "Epoch": epoch, "Train Accuracy": train_accuracy,
                       "Validation Accuracy": valid_accuracy, "Loss": epoch_loss})

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "loss": epoch_loss / len(train_loader)
            }
            metrics["epochs"].append(epoch_metrics)

        print(f'Client : {client_idx} training complete.')

        print(f'Testing  {client_idx}')
        test_accuracy = self.evaluate(model, self.test_dataloader)
        test_accuracies.append(test_accuracy)
        test_metrics = {
            "epoch": self.args.epochs ,
            "test_accuracy": test_accuracy
        }
        metrics["test_metrics"].append(test_metrics)

        wandb.log({"Client": client_idx, "Test Accuracy": test_accuracy})
        print(f'Client: {client_idx}, Test accuracy : {test_accuracy}')


        with open(f'client_{client_idx}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        return train_accuracies, valid_accuracies, test_accuracies

    def evaluate(self, model, loader):
        model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy


if __name__ == "__main__":
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the content root directory
    content_root = os.path.dirname(script_dir)

    # Get the source root directory
    source_root = os.path.join(
        content_root, "models/"
    )  # Replace 'src' with your source root directory name

    # Add the content root and source root to PYTHONPATH
    sys.path.append(content_root)
    sys.path.append(source_root)
    if not os.path.exists('pretrained'):
        os.mkdir('pretrained')
    if not os.path.exists('reports'):
        os.mkdir('reports')
    start_time = time.time()
    args = args_parser()

    # call client
    client = ClientTrainer()
    client.train_and_evaluate()

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
