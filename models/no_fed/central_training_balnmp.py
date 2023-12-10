import json
import sys
import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import pandas as pd
from balnmp import MILNetWithClinicalData
from tqdm import tqdm
from args_parser import args_parser
from PIL import Image
import time

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


class Trainer:
    def __init__(self):
        self.args = args_parser()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.train_dataset, self.test_dataset = self.get_datasets()
        self.model = MILNetWithClinicalData(num_classes=2, backbone_name="vgg16_bn")

    def get_datasets(self):
        # Implementation of dataset loading (BreastDataset)
        data_dir = "../data/balnmp/"
        balnmp_data_type = self.args.balnmp_data_type
        json_train_path = os.path.join(data_dir, 'json/train-' + balnmp_data_type + '.json')
        json_test_path = os.path.join(data_dir, 'json/test-' + balnmp_data_type + '.json')

        with open(json_train_path) as f:
            json_train_data = json.load(f)
        with open(json_test_path) as f:
            json_test_data = json.load(f)

        train_dataset = BreastDataset(json_train_data, data_dir, os.path.join(data_dir,
                                                                              'clinical_data/preprocessed-' + balnmp_data_type + '.xlsx'),
                                      is_preloading=True)
        test_dataset = BreastDataset(json_test_data, data_dir,
                                     os.path.join(data_dir, 'clinical_data/preprocessed-' + balnmp_data_type + '.xlsx'),
                                     is_preloading=True)

        return train_dataset, test_dataset

    def train_and_evaluate(self):
        train_loader, valid_loader = self.prepare_dataloaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = {"epochs": [], "test_metrics": []}
        self.model.to(self.device)

        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss = 0
            for item in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"):
                bag_tensor, label = item["bag_tensor"].to(self.device), torch.tensor([item["label"]],
                                                                                     dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                clinical_data = None
                if "clinical_data" in item:
                    clinical_data = torch.tensor(item["clinical_data"], dtype=torch.float32).to(self.device)
                clinical_data, aggregated_feature, attention = self.model(bag_tensor, clinical_data)
                fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.model.expand_times).float()],
                                       dim=-1)
                output = self.model.classifier(fused_data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_accuracy = self.evaluate(valid_loader)
            print(f'Epoch: {epoch}, Loss: {epoch_loss}, Val Acc: {val_accuracy}')
            metrics["epochs"].append({"epoch": epoch, "val_accuracy": val_accuracy, "loss": epoch_loss})

            if (epoch + 1) % 5 == 0:
                test_accuracy = self.evaluate(self.test_dataset)
                print(f'Test Accuracy: {test_accuracy}')
                metrics["test_metrics"].append({"epoch": epoch, "test_accuracy": test_accuracy})

        print("Training complete.")
        with open('training_metrics_balnmp_centralized_100.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def prepare_dataloaders(self):
        train_size = int(0.8 * len(self.train_dataset))
        valid_size = len(self.train_dataset) - train_size
        train_dataset, valid_dataset = random_split(self.train_dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.test_dataset_bs, shuffle=False)
        return train_loader, valid_loader

    def evaluate(self, loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for item in loader:
                bag_tensor, label = item["bag_tensor"].to(self.device), torch.tensor([item["label"]],
                                                                                     dtype=torch.long).to(self.device)
                clinical_data = None
                if "clinical_data" in item:
                    clinical_data = torch.tensor(item["clinical_data"], dtype=torch.float32).to(self.device)
                clinical_data, aggregated_feature, attention = self.model(bag_tensor, clinical_data)
                fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.model.expand_times).float()],
                                       dim=-1)
                output = self.model.classifier(fused_data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
        return accuracy


if __name__ == "__main__":
    # time.sleep(21600)
    start_time = time.time()
    trainer = Trainer()
    trainer.train_and_evaluate()
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
