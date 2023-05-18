import copy
import os
import pickle
import time
from typing import List

import federated_clients
import numpy as np
import server
import torch
from args_parser import args_parser
from nn_models import MnistModel, ResNetModel
from tensorboardX import SummaryWriter
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


class Client:
    def __init__(self) -> None:
        # define paths
        path_project = os.path.abspath("..")
        self.logger = SummaryWriter("../logs")  # instantiate a TensorBoard logger

        self.args = args_parser()  # parse command line arguments
        exp_details(
            self.args
        )  # print some details about the experiment, like dataset, model, etc.

        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        self.device = "cuda" if self.args.gpu else "cpu"

        # load the dataset and divide it into user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

        # BUILD MODEL
        if self.args.dataset == "mnist":
            self.global_model = MnistModel()
        elif self.args.dataset == "cifar":
            self.global_model = ResNetModel()
        else:
            exit("Error: unrecognized model")

        # create a non-encrypted copy of the global model
        self.global_model_non_encry = copy.deepcopy(self.global_model)
        self.time_taken_each_round = []
        # initialize server
        self.server = server_classes[
            self.args.fed_algo
        ]  # instantiate the server class corresponding to the selected federated algorithm
        self.client = client_classes[self.args.fed_algo](
            self.args, self.train_dataset, self.user_groups, self.logger, self.server
        )  # instantiate the client class corresponding to the selected federated algorithm

    # def get_client_weights(self):
    #     if self.args.client_weights:
    #         return self.args.client_weights
    #     else:
    #         return [len(value) for user, value in self.user_groups.items()]

    # def aggregate_by_weights(self, local_weights):
    #     # update global weights without encryption
    #     if self.args.train_without_encryption:
    #         global_weights = self.server.average_weights_nonencrypted(local_weights)
    #         self.global_model_non_encry.load_state_dict(global_weights)
    #     # update global weights with encryption
    #     shapes = {}
    #     for key, value in local_weights[0].items():
    #         shapes[key] = value.shape
    #
    #     # TODO: make use of context in averaging after serializing input and context
    #     encrypted_weights = self.he.encrypt_client_weights(local_weights)
    #
    #     if self.args.fed_algo == 'WeightedFedAvg':
    #         # using server class method to encrypt weights with client weights
    #         client_weights = self.get_client_weights()  # can be number of items in each client or any other weight
    #         global_averaged_encrypted_weights = self.server((encrypted_weights, client_weights)).aggregate()
    #         global_weights = self.he.decrypt_and_average_weights(global_averaged_encrypted_weights, shapes,
    #                                                              client_weights=sum(client_weights),
    #                                                              secret_key=self.secret_key)
    #     else:
    #         # using server class method to encrypt weights without client weights
    #         global_averaged_encrypted_weights = self.server(encrypted_weights).aggregate()
    #         global_weights = self.he.decrypt_and_average_weights(global_averaged_encrypted_weights, shapes,
    #                                                              client_weights=self.args.num_clients,
    #                                                              secret_key=self.secret_key)
    #
    #     # update global weights
    #     self.global_model.load_state_dict(global_weights)

    # def aggregate_by_int_gradients(self, temp_model):
    #     bound = 2 ** 3
    #     prec = 32
    #     params_modules = list(temp_model.named_parameters())
    #     params_grad_list = []
    #     for params_module in params_modules:
    #         name, params = params_module
    #         params_grad_list.append(copy.deepcopy(params.grad).view(-1))
    #
    #     params_grad = ((torch.cat(params_grad_list, 0) + bound) * 2 ** prec).long()

    # def aggregate_by_gradients(self, temp_model):
    #     bound = 2 ** 3
    #     prec = 32
    #     params_modules = list(temp_model.named_parameters())
    #     params_grad_list = []
    #     for params_module in params_modules:
    #         name, params = params_module
    #         params_grad_list.append(copy.deepcopy(params.grad).view(-1))
    #
    #     params_grad = ((torch.cat(params_grad_list, 0) + bound) * 2 ** prec).long()

    # def train_model(self, test=False, save=False, plot=False):
    #     # Set the model to train and send it to device.
    #     self.global_model.to(self.device)
    #     self.global_model.train()
    #
    #     if self.args.train_without_encryption:
    #         self.global_model_non_encry.to(self.device)
    #         self.global_model_non_encry.train()
    #
    #     print(self.global_model)
    #
    #     # copy weights
    #     self.global_weights = self.global_model.state_dict()
    #
    #     # Training
    #     train_loss, train_accuracy = [], []
    #     print_every = 2
    #
    #     for round_ in tqdm(range(self.args.rounds)):
    #         local_weights, local_losses = [], []
    #         print(f'\n | Global Training Round : {round_ + 1} |\n')
    #
    #         self.global_model.train()
    #         m = self.args.num_clients
    #
    #         idxs_users = range(m)
    #         if self.args.aggregation_type == 'weights':
    #             for idx in idxs_users:
    #                 local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
    #                                           idxs=self.user_groups[idx], logger=self.logger)
    #                 model, loss = local_model.update_weights(
    #                     model=copy.deepcopy(self.global_model), global_round=round_)
    #                 local_weights.append(copy.deepcopy(model.state_dict()))
    #                 local_losses.append(copy.deepcopy(loss))
    #             self.aggregate_by_weights(local_weights)
    #         elif self.args.aggregation_type == 'gradients':
    #             for idx in idxs_users:
    #                 local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
    #                                           idxs=self.user_groups[idx], logger=self.logger)
    #                 model, loss = local_model.update_weights(
    #                     model=copy.deepcopy(self.global_model), global_round=round_)
    #                 local_weights.append(copy.deepcopy(model.state_dict()))
    #                 local_losses.append(copy.deepcopy(loss))
    #             self.aggregate_by_gradients(model)
    #
    #         loss_avg = sum(local_losses) / len(local_losses)
    #         train_loss.append(loss_avg)
    #
    #         # Calculate avg training accuracy over all users at every epoch
    #         list_acc, list_loss = self.evaluate_model(idx)
    #
    #         train_accuracy.append(sum(list_acc) / len(list_acc))
    #
    #         # print global training loss after every 'i' rounds
    #         if (round_ + 1) % print_every == 0:
    #             print(f' \nAvg Training Stats after {round_ + 1} global rounds:')
    #             print(f'Training Loss : {np.mean(np.array(train_loss))}')
    #             print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
    #
    #     if test:
    #         self.test_model(train_accuracy)
    #     if save:
    #         self.save_model(train_loss, train_accuracy)
    #     if plot:
    #         self.plot(train_loss, train_accuracy)

    def train_clients(
        self, test: bool = False, save: bool = False, plot: bool = False
    ) -> None:
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
            print(f"\n | Global Training Round : {round_ + 1} |\n")

            self.global_model.train()

            # Call the client's train_clients method to perform local training and aggregation.
            (
                train_loss,
                train_acc,
                global_weights,
                self.global_model_non_encry,
                time_taken,
            ) = self.client.train_clients(
                self.global_model, round_, self.global_model_non_encry
            )

            self.global_model.load_state_dict(copy.deepcopy(global_weights))

            train_accuracy_list.append(train_acc)
            train_loss_list.append(train_loss)
            self.time_taken_each_round.append(time_taken)
            if (round_ + 1) % print_every == 0:
                print(f" \nAvg Training Stats after {round_ + 1} global rounds:")
                print(f"Training Loss : {np.mean(np.array(train_loss_list[-1]))}")
                print(
                    "Train Accuracy: {:.2f}% \n".format(100 * train_accuracy_list[-1])
                )

        # If specified, evaluate/save/plot the trained model
        if test:
            self.test_model(train_accuracy_list)

        if save:
            self.save_model(train_loss_list, train_accuracy_list)

        if plot:
            self.plot(train_loss_list, train_accuracy_list)

    def test_model(self, train_accuracy: List[float]) -> None:
        # Test inference after completion of training
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        test_acc, test_loss = self.client.evaluate_model(self.global_model, test_loader)

        print(
            f" \n Results after {self.args.rounds} global rounds of training on encrypted model:"
        )
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        if self.args.train_without_encryption:
            # Test inference on non encrypted model after completion of training
            test_acc_nonencry, test_loss_nonencr = self.client.evaluate_model(
                self.global_model_non_encry, test_loader
            )

            print(
                f" \n Results after {self.args.rounds} global rounds of training on non-encrypted model:"
            )
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc_nonencry))

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

        # Get the keys and values from the dictionary
        phases = list(self.time_taken_each_round[0].keys())
        times = list(self.time_taken_each_round[0].values())

        # Create a bar chart
        plt.bar(phases, times)

        # Set the x and y labels
        plt.xlabel("Phases")
        plt.ylabel("Time Required")

        # Set the title
        plt.title("Time Required by Each Phase")
        plt.savefig(
            "../reports/fed_{}_{}_{}_iid[{}]_E[{}]_B[{}]_time.png".format(
                self.args.fed_algo,
                self.args.dataset,
                self.args.rounds,
                self.args.iid,
                self.args.epochs,
                self.args.local_bs,
            )
        )
