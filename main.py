import time

import torch
from models.args_parser import args_parser
from models.client import Client
import sys
import os


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
    client = Client()
    client.train_clients(test=True, save=True, plot=True)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
