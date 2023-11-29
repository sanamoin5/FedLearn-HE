import wandb
from tqdm import tqdm
from args_parser import args_parser
import sys

import numpy as np
import copy
import time

import torch.utils.data.dataloader
from balnmp import MILNetWithClinicalData
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import torchvision
import json
import pandas as pd
from torch.nn import functional as F
from sklearn import metrics

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),  # resize to 224*224
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # normalization
    ]
)
to_tensor = torchvision.transforms.ToTensor()
Image.MAX_IMAGE_PIXELS = None


class BreastDataset(torch.utils.data.Dataset):
    """Pytorch dataset api for loading patches and preprocessed clinical data of breast."""

    def __init__(self, json_data, data_dir_path='data/balnmp', clinical_data_path=None, is_preloading=True):
        self.data_dir_path = data_dir_path
        self.is_preloading = is_preloading

        if clinical_data_path is not None:
            print(f"load clinical data from {clinical_data_path}")
            self.clinical_data_df = pd.read_excel(clinical_data_path, index_col="p_id", engine="openpyxl")
        else:
            self.clinical_data_df = None

        self.json_data = json_data

        if self.is_preloading:
            self.bag_tensor_list = self.preload_bag_data()

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        label = int(self.json_data[index]["label"])
        patient_id = self.json_data[index]["id"]
        patch_paths = self.json_data[index]["patch_paths"]

        data = {}
        if self.is_preloading:
            data["bag_tensor"] = self.bag_tensor_list[index]
        else:
            data["bag_tensor"] = self.load_bag_tensor(
                [os.path.join(self.data_dir_path, p_path) for p_path in patch_paths])

        if self.clinical_data_df is not None:
            data["clinical_data"] = self.clinical_data_df.loc[int(patient_id)].to_numpy()

        data["label"] = label
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths

        return data

    def preload_bag_data(self):
        """Preload data into memory"""

        bag_tensor_list = []
        for item in tqdm(self.json_data, ncols=120, desc="Preloading bag data"):
            patch_paths = [os.path.join(self.data_dir_path, p_path) for p_path in item["patch_paths"]]
            bag_tensor = self.load_bag_tensor(patch_paths)  # [N, C, H, W]
            bag_tensor_list.append(bag_tensor)

        return bag_tensor_list

    def load_bag_tensor(self, patch_paths):
        """Load a bag data as tensor with shape [N, C, H, W]"""

        patch_tensor_list = []
        for p_path in patch_paths:
            patch = Image.open(p_path).convert("RGB")
            patch_tensor = transform(patch)  # [C, H, W]
            patch_tensor = torch.unsqueeze(patch_tensor, dim=0)  # [1, C, H, W]
            patch_tensor_list.append(patch_tensor)

        bag_tensor = torch.cat(patch_tensor_list, dim=0)  # [N, C, H, W]

        return bag_tensor


class DatasetSplit2(torch.utils.data.Dataset):
    """Custom dataset class for splitting a dataset based on indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.dataset[original_index]






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

        self.model = MILNetWithClinicalData(num_classes=2, backbone_name="vgg16_bn")

    def train_val(self, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        idxs_train = idxs[: int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):]

        trainloader = DataLoader(
            DatasetSplit2(self.dataset, idxs_train),
            batch_size=1,
            shuffle=True,
        )
        validloader = DataLoader(
            DatasetSplit2(self.dataset, idxs_val),
            batch_size=1,
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

    def get_datasets_and_user_groups(self, num_clients):
        train_dataset, test_dataset, user_groups = None, None, None
        if args.model == "balnmp":
            data_dir = "../data/balnmp/"
            print(os.getcwd())
            print('-----------')
            balnmp_data_type = args.balnmp_data_type
            json_train_path = os.path.join(data_dir, 'json/train-' + balnmp_data_type + '.json')
            json_test_path = os.path.join(data_dir, 'json/test-' + balnmp_data_type + '.json')
            json_val_path = os.path.join(data_dir, 'json/val-' + balnmp_data_type + '.json')

            with open(json_train_path) as f:
                json_train_data = json.load(f)
            with open(json_val_path) as f:
                json_val_data = json.load(f)

            json_train_val_data = json_train_data + json_val_data

            with open(json_test_path) as f:
                json_test_data = json.load(f)

            train_dataset = BreastDataset(json_train_val_data, data_dir,
                                          os.path.join(data_dir, 'clinical_data/preprocessed-' +
                                                       balnmp_data_type + '.xlsx'), is_preloading=True)

            test_dataset = BreastDataset(json_test_data, data_dir,
                                         os.path.join(data_dir, 'clinical_data/preprocessed-' +
                                                      balnmp_data_type + '.xlsx'), is_preloading=True)

            # sample training data amongst users
            if args.iid:
                user_groups = self.dataset_iid(train_dataset, args.num_clients)

        return train_dataset, test_dataset, user_groups

    def train_and_evaluate(self):
        all_train_accuracies, all_valid_accuracies, test_accuracies = [], [], []

        for client_idx in range(self.args.num_clients):
            print(f"Training client {client_idx}")
            client_model = copy.deepcopy(self.model.to(self.device))
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            train_accuracies, valid_accuracies, test_accuracies = self.train(client_model, optimizer, criterion,
                                                                             client_idx)
            all_train_accuracies.append(train_accuracies)
            all_valid_accuracies.append(valid_accuracies)

        return all_train_accuracies, all_valid_accuracies, test_accuracies

    def train(self, model, optimizer, criterion, client_idx):

        model.train()
        metrics = {"epochs": [], "test_metrics": []}

        train_loader, valid_loader = self.train_val(list(self.user_groups[client_idx]))

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=float(self.args.lr), momentum=float(self.args.momentum)
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(self.args.lr), weight_decay=float(self.args.weight_decay)
            )
        total_loss = 0
        label_list = []
        score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
        id_list = []
        patch_path_list = []
        attention_value_list = []
        train_accuracies, valid_accuracies, test_accuracies = [], [], []
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}, Client {client_idx}")
            for index, item in enumerate(train_loader, start=1):
                bag_tensor, label = item["bag_tensor"].to(self.device), item["label"].to(self.device)
                clinical_data = torch.tensor(item["clinical_data"][0], dtype=torch.float32).to(self.device) if "clinical_data" in item else None

                optimizer.zero_grad()
                clinical_data, aggregated_feature, attention = model(bag_tensor, clinical_data)
                fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, model.expand_times).float()],
                                       dim=-1)
                output = model.classifier(fused_data)

                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(train_loader))

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[
                    1].cpu().item()  # use the predicted positive probability as score
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention[0].cpu().tolist())

            if self.args.merge_method != "not_use":
                id_list, label_list, score_list, bag_num_list = self.merge_result(id_list, label_list, score_list,
                                                                                  self.args.merge_method)

            if (epoch + 1) % 5 == 0:
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

            average_loss = epoch_loss / len(train_loader)
            predicted_label_list = [1 if score >= 0.5 else 0 for score in score_list]
            accuracy = sum(pred == act for pred, act in zip(predicted_label_list, label_list)) / len(label_list)

            # Evaluate on training and validation set after each epoch
            train_accuracy = self.evaluate(model, train_loader)
            valid_accuracy = self.evaluate(model, valid_loader)
            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)
            print(
                f'Epoch: {epoch}, Client: {client_idx}, Loss: {epoch_loss}, Train Acc: {train_accuracy}, Val Acc: {valid_accuracy}')

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
            "epoch": self.args.epochs,
            "test_accuracy": test_accuracy
        }
        metrics["test_metrics"].append(test_metrics)

        wandb.log({"Client": client_idx, "Test Accuracy": test_accuracy})
        print(f'Client: {client_idx}, Test accuracy : {test_accuracy}')

        with open(f'client_{client_idx}_balnmp_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        return train_accuracies, valid_accuracies, test_accuracies

    def merge_result(self, id_list, label_list, score_list, method):
        """Merge predicted results of all bags for each patient"""

        assert method in ["max", "mean"]
        merge_method = np.max if method == "max" else np.mean

        df = pd.DataFrame()
        df["id"] = id_list
        df["label"] = label_list
        df["score"] = score_list
        # https://www.jb51.cc/python/438695.html
        df = df.groupby(by=["id", "label"])["score"].apply(list).reset_index()
        df["bag_num"] = df["score"].apply(len)
        df["score"] = df["score"].apply(merge_method, args=(0,))

        return df["id"].tolist(), df["label"].tolist(), df["score"].tolist(), df["bag_num"].tolist()

    def evaluate(self, model, testloader):
        """Returns the inference accuracy and loss."""
        total_loss = 0
        label_list = []
        score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
        id_list = []
        patch_path_list = []
        attention_value_list = []
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train(False)
        with torch.no_grad():
            for index, item in enumerate(testloader, start=1):
                bag_tensor, label = item["bag_tensor"].to(self.device), item["label"].to(self.device)
                clinical_data = torch.tensor(item["clinical_data"][0], dtype=torch.float32).to(self.device) if "clinical_data" in item else None
                clinical_data, aggregated_feature, attention = model(bag_tensor, clinical_data)
                fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, model.expand_times).float()],
                                       dim=-1)
                # feature fusion
                output = model.classifier(fused_data)

                # output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[
                    1].cpu().item()  # use the predicted positive probability as score
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention[0].cpu().tolist())

        if self.args.merge_method != "not_use":
            id_list, label_list, score_list, bag_num_list = self.merge_result(id_list, label_list, score_list,
                                                                              self.args.merge_method)

        average_loss = total_loss / len(testloader)
        predicted_label_list = [1 if score >= 0.5 else 0 for score in score_list]
        accuracy = sum(pred == act for pred, act in zip(predicted_label_list, label_list)) / len(label_list)

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
