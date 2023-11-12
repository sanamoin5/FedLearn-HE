import torch
from .FedClient import measure_time, DatasetSplit2
from .BatchCryptFedClient import BatchCryptBasedFedAvgClient
from .FedAvgClientWithoutEncryption import FedAvgClientWithoutEncryption
from .FedAvgClient import FedAvgClient
from .GradientBasedFedAvgClient import GradientBasedFedAvgClient
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class MultiModalPreFusionFedClient(FedAvgClientWithoutEncryption):
    def __init__(self, args, train_dataset, user_groups, json_logs, server):
        super().__init__(args, train_dataset, user_groups, json_logs, server)
        self.args = args
        self.print_every = 2

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

    @measure_time
    def train_local_model(self, model, global_round, client_idx):
        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=float(self.args.lr), momentum=float(self.args.momentum)
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=float(self.args.lr), weight_decay=float(self.args.weight_decay)
            )
        total_loss = 0
        label_list = []
        score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
        id_list = []
        patch_path_list = []
        attention_value_list = []

        model.train()

        for index, item in enumerate(self.train_loader[client_idx], start=1):
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            self.optimizer.zero_grad()
            clinical_data, aggregated_feature, attention = model(bag_tensor, clinical_data)

            fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, model.expand_times).float()], dim=-1)
            # feature fusion
            output = model.classifier(fused_data)

            loss = F.cross_entropy(output, label)
            loss.backward()
            self.optimizer.step()
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

            average_loss = total_loss / len(self.train_loader[client_idx])
            predicted_label_list = [1 if score >= 0.5 else 0 for score in score_list]
            accuracy = sum(pred == act for pred, act in zip(predicted_label_list, label_list)) / len(label_list)

            if self.args.verbose and (index % 10 == 0):
                print(
                    "| Global Round : {} | Client : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        global_round,
                        client_idx,
                        index * len(bag_tensor),
                        len(self.train_loader[client_idx].dataset),
                        100.0 * index / len(self.train_loader[client_idx]),
                        loss.item(),
                    )
                )

        return model, total_loss

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

    def evaluate_model(self, model, testloader):
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
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None
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

        return accuracy, average_loss
