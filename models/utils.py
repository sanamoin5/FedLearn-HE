#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from argparse import Namespace
from typing import Dict, Set, Tuple

from numpy import int64
from models.sampling import (
    dataset_iid,
    cifar_noniid,
    mnist_noniid,
    mnist_noniid_unequal,
)
from torchvision import datasets, transforms

import os
from PIL import Image
import torch
import torchvision
import tqdm
import json
import pandas as pd

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

    def __init__(self, json_path, data_dir_path='data/balnmp', clinical_data_path=None, is_preloading=True):
        self.data_dir_path = data_dir_path
        self.is_preloading = is_preloading

        if clinical_data_path is not None:
            print(f"load clinical data from {clinical_data_path}")
            self.clinical_data_df = pd.read_excel(clinical_data_path, index_col="p_id", engine="openpyxl")
        else:
            self.clinical_data_df = None

        with open(json_path) as f:
            print(f"load data from {json_path}")
            self.json_data = json.load(f)

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
            data["bag_tensor"] = self.load_bag_tensor([os.path.join(self.data_dir_path, p_path) for p_path in patch_paths])

        if self.clinical_data_df is not None:
            data["clinical_data"] = self.clinical_data_df.loc[int(patient_id)].to_numpy()

        data["label"] = label
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths

        return data

    def preload_bag_data(self):
        """Preload data into memory"""

        bag_tensor_list = []
        for item in tqdm.tqdm(self.json_data, ncols=120, desc="Preloading bag data"):
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

def get_dataset(args):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == "cifar":
        data_dir = "data/cifar/"
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = dataset_iid(train_dataset, args.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_clients)

    elif args.dataset == "mnist":
        data_dir = "data/mnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = dataset_iid(train_dataset, args.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_clients)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_clients)

    elif args.dataset == "balnmp":
        data_dir = "data/balnmp/"
        train_dataset = BreastDataset(os.path.join(data_dir, 'json/train-type-3.json'), data_dir, os.path.join(data_dir,'clinical_data/preprocessed-type-3.xlsx'),
                                      is_preloading=False)

        val_dataset = BreastDataset(os.path.join(data_dir, 'json/val-type-3.json'), data_dir, os.path.join(data_dir,'clinical_data/preprocessed-type-3.xlsx'),
                                    is_preloading=False)

        test_dataset = BreastDataset(os.path.join(data_dir, 'json/test-type-3.json'), data_dir, os.path.join(data_dir,'clinical_data/preprocessed-type-3.xlsx'),
                                     is_preloading=False)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = dataset_iid(train_dataset, args.num_clients)
            user_groups_val = dataset_iid(val_dataset, args.num_clients)
        else:
            # Sample Non-IID user data from balnmp
            pass
        return (train_dataset, val_dataset), test_dataset, (user_groups_train, user_groups_val)

    return train_dataset, test_dataset, user_groups


def exp_details(args: Namespace) -> None:
    print("\nExperimental details:")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.rounds}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"Federated learning algorithm: {args.fed_algo}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.epochs}\n")
    print(f"    Num Clients       : {args.num_clients}\n")

    print("    Homomorphic Encryption parameters:")
    print(f"    Scheme name       : {args.he_scheme_name}\n")

    return

from torch import nn
import torchvision


BACKBONES = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152"
]


class BackboneBuilder(nn.Module):
    """Build backbone with the last fc layer removed"""

    def __init__(self, backbone_name):
        super().__init__()

        assert backbone_name in BACKBONES

        complete_backbone = torchvision.models.__dict__[backbone_name](pretrained=True)

        if backbone_name.startswith("vgg"):
            assert backbone_name in ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            self.extractor, self.output_features_size = self.vgg(complete_backbone)
        elif backbone_name.startswith("resnet"):
            assert backbone_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
            self.extractor, self.output_features_size = self.resnet(backbone_name, complete_backbone)
        else:
            raise NotImplementedError

    def forward(self, x):
        patch_features = self.extractor(x)

        return patch_features

    def vgg(self, complete_backbone):
        output_features_size = 512 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def resnet(self, backbone_name, complete_backbone):
        if backbone_name in ["resnet18", "resnet34"]:
            output_features_size = 512 * 1 * 1
        else:
            output_features_size = 2048 * 1 * 1
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
