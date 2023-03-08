#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from args_parser import args_parser
from update import test_inference
from nn_models import ResNetModel, MnistModel
from client import client

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client()
    # call client, run federated way

    # call standalone way
