import json
from args_parser import args_parser
import sys
import os
import time

import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from cifar_resnet import CIFAR10_ResNet


def load_data(args):
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

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    # Split train dataset into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_dataset_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_dataset_bs, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def train_and_evaluate():
    args = args_parser()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data(args)
    model = CIFAR10_ResNet.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = {"epochs": [], "test_metrics": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader, device)
        print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_accuracy": val_accuracy
        }
        metrics["epochs"].append(epoch_metrics)

        # Evaluate on test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_accuracy = evaluate(model, test_loader, device)
            print(f'Epoch: {epoch}, Test Accuracy: {test_accuracy}')
            test_metrics = {
                "epoch": epoch + 1,
                "test_accuracy": test_accuracy
            }
            metrics["test_metrics"].append(test_metrics)

    print("Training complete.")
    with open('training_metrics_cnn_centralized.json', 'w+') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    train_and_evaluate()
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
