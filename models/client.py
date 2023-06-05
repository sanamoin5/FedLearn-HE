import copy
import json
import logging
import pickle
import time
from typing import List

import numpy as np
import server
import torch
import wandb
from args_parser import args_parser
from models import federated_clients
from nn_models import MnistModel, ResNetModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import exp_details, get_dataset


server_classes = {
    "FedAvg": server.FedAvgServer,
    "WeightedFedAvg": server.FedAvgServer,
    "GradientBasedFedAvg": server.FedAvgServer,
    "QuantFedClient": server.FedAvgServer,
    "BatchCryptClient": server.FedAvgServer,
}
client_classes = {
    "FedAvg": federated_clients.FedAvgClient,
    "WeightedFedAvg": federated_clients.WeightedFedAvgClient,
    "GradientBasedFedAvg": federated_clients.GradientBasedFedAvgClient,
    "QuantFedClient": federated_clients.QuantFedClient,
    "BatchCryptClient": federated_clients.BatchCryptBasedFedAvgClient,
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Client:
    def __init__(self) -> None:
        wandb.init(project="FedLearn-HE")
        self.args = args_parser()  # parse command line arguments
        # BUILD MODEL
        if self.args.dataset == "mnist":
            self.global_model = MnistModel()
        elif self.args.dataset == "cifar":
            self.global_model = ResNetModel()
        else:
            exit("Error: unrecognized model")

        exp_details(
            self.args
        )  # print some details about the experiment, like dataset, model, etc.
        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        self.device = "cuda" if self.args.gpu else "cpu"
        self.json_logs = {}

        # load the dataset and divide it into user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

        # create a non-encrypted copy of the global model
        self.global_model_non_encry = copy.deepcopy(self.global_model)
        self.time_taken_each_round = []
        # initialize server
        self.server = server_classes[
            self.args.fed_algo
        ]  # instantiate the server class corresponding to the selected federated algorithm
        self.client = client_classes[self.args.fed_algo](
            self.args, self.train_dataset, self.user_groups, self.json_logs, self.server
        )  # instantiate the client class corresponding to the selected federated algorithm

    def train_clients(self, test: bool = False, save: bool = False, plot: bool = False):
        self.global_model.to(self.device)
        self.global_model.train()

        # If specified, also set the non-encrypted model to train
        if self.args.train_without_encryption:
            self.global_model_non_encry.to(self.device)
            self.global_model_non_encry.train()

        # Initialize the global weights to the current state of the global model.
        self.global_weights = self.global_model.state_dict()

        train_loss_list, train_accuracy_list = [], []

        print_every = 2

        # Loop through the specified number of rounds of training.
        for round_ in tqdm(range(self.args.rounds)):
            logging.info(f"\n | Global Training Round : {round_ + 1} |\n")

            self.global_model.train()

            # Call the client's train_clients method to perform local training and aggregation.
            (
                train_loss,
                train_acc,
                global_weights,
                self.global_model_non_encry,
            ) = self.client.train_clients(
                self.global_model, round_, wandb, self.global_model_non_encry
            )

            self.global_model.load_state_dict(copy.deepcopy(global_weights))

            train_accuracy_list.append(train_acc)
            train_loss_list.append(train_loss)
            wandb.log({"round_accuracy": train_acc, "round_loss": train_loss})
            self.update_json_logs_list("round_accuracy", train_acc)
            self.update_json_logs_list("round_loss", train_loss)

            if (round_ + 1) % print_every == 0:
                logging.info(f" \nAvg Training Stats after {round_ + 1} global rounds:")
                logging.info(
                    f"Training Loss : {np.mean(np.array(train_loss_list[-1]))}"
                )
                logging.info(
                    "Train Accuracy: {:.2f}% \n".format(100 * train_accuracy_list[-1])
                )

        # If specified, evaluate/save/plot the trained model
        if test:
            self.test_model(train_accuracy_list)

        if save:
            self.save_model(train_loss_list, train_accuracy_list)

        if plot:
            self.plot(train_loss_list, train_accuracy_list)

        self.update_wandb_exp(federated_clients.FedClient.function_times)

        wandb.finish()
        try:
            with open("json_logs.json", "r") as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []
        existing_data.append(self.json_logs)
        with open("json_logs.json", "w+") as json_file:
            json.dump(existing_data, json_file)

    def test_model(self, train_accuracy: List[float]) -> None:
        # Test inference after completion of training
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        test_acc, test_loss = self.client.evaluate_model(self.global_model, test_loader)

        logging.info(
            f" \n Results after {self.args.rounds} global rounds of training on encrypted model:"
        )
        logging.info(
            "|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1])
        )
        logging.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        if self.args.train_without_encryption:
            # Test inference on non encrypted model after completion of training
            test_acc_nonencry, test_loss_nonencr = self.client.evaluate_model(
                self.global_model_non_encry, test_loader
            )

            logging.info(
                f" \n Results after {self.args.rounds} global rounds of training on non-encrypted model:"
            )
            logging.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc_nonencry))

            wandb.log(
                {
                    "test_acc_nonencry": test_acc_nonencry,
                    "test_loss_nonencr": test_loss_nonencr,
                }
            )
            self.json_logs["test_acc_nonencry"] = test_acc_nonencry
            self.json_logs["test_loss_nonencr"] = test_loss_nonencr

        wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
        self.json_logs["test_accuracy"] = test_acc
        self.json_logs["test_loss"] = test_loss

    def save_model(self, train_loss: List[float], train_accuracy: List[float]) -> None:
        # Saving the objects train_loss and train_accuracy:
        file_name = "../pretrained/{}_{}_iid[{}]_E[{}]_B[{}].pkl".format(
            self.args.dataset,
            self.args.rounds,
            self.args.iid,
            self.args.epochs,
            self.args.local_bs,
        )
        time.sleep(3)
        with open(file_name, "wb") as f:
            pickle.dump([train_loss, train_accuracy], f)

    def plot(self, train_loss: List[float], train_accuracy: List[float]) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        # Plot Loss curve
        plt.figure()
        plt.title("Training Loss vs Communication rounds")
        plt.plot(range(len(train_loss)), train_loss, color="r")
        plt.ylabel("Training loss")
        plt.xlabel("Communication Rounds")
        plt.savefig(
            "../reports/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png".format(
                self.args.fed_algo,
                self.args.dataset,
                self.args.rounds,
                self.args.iid,
                self.args.epochs,
                self.args.local_bs,
            )
        )

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title("Average Accuracy vs Communication rounds")
        plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
        plt.ylabel("Average Accuracy")
        plt.xlabel("Communication Rounds")
        plt.savefig(
            "../reports/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_acc.png".format(
                self.args.fed_algo,
                self.args.dataset,
                self.args.rounds,
                self.args.iid,
                self.args.epochs,
                self.args.local_bs,
            )
        )
        # Plot Time vs Phase
        plt.figure()

    def update_wandb_exp(self, recorded_times):
        logging.info("LOGGING..")

        wandb.config.device = self.device
        wandb.config.optimizer = self.args.optimizer
        wandb.config.lr = self.args.lr
        wandb.config.global_rounds = self.args.rounds
        wandb.config.is_iid = self.args.iid
        wandb.config.fed_algo = self.args.fed_algo
        wandb.config.local_batch_size = self.args.local_bs
        wandb.config.local_epochs = self.args.epochs
        wandb.config.num_clients = self.args.num_clients
        wandb.config.he_scheme_name = self.args.he_scheme_name

        self.json_logs["device"] = self.device
        self.json_logs["optimizer"] = self.args.optimizer
        self.json_logs["lr"] = self.args.lr
        self.json_logs["global_rounds"] = self.args.rounds
        self.json_logs["is_iid"] = self.args.iid
        self.json_logs["fed_algo"] = self.args.fed_algo
        self.json_logs["local_batch_size"] = self.args.local_bs
        self.json_logs["local_epochs"] = self.args.epochs
        self.json_logs["num_clients"] = self.args.num_clients
        self.json_logs["he_scheme_name"] = self.args.he_scheme_name

        for key, value in recorded_times.items():
            wandb.log({key: value})
            self.json_logs[key] = value

    def update_json_logs_list(self, key, value):
        if key in self.json_logs:
            self.json_logs[key].append(value)
        else:
            self.json_logs[key] = [value]
