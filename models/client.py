import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from args_parser import args_parser
from utils import get_dataset, exp_details
from nn_models import ResNetModel, MnistModel
import server
import federated_clients
from homomorphic_encryption import HEScheme

server_classes = {'FedAvg': server.FedAvgServer, 'WeightedFedAvg': server.WeightedFedAvgServer}
client_classes = {'FedAvg': federated_clients.FedAvgClient}


class Client:
    def __init__(self):
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')  # instantiate a TensorBoard logger

        self.args = args_parser()  # parse command line arguments
        exp_details(self.args)  # print some details about the experiment, like dataset, model, etc.

        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # load the dataset and divide it into user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

        # BUILD MODEL
        if self.args.dataset == 'mnist':
            self.global_model = MnistModel()
        elif self.args.dataset == 'cifar':
            self.global_model = ResNetModel()
        else:
            exit('Error: unrecognized model')

        # create a non-encrypted copy of the global model
        self.global_model_non_encry = copy.deepcopy(self.global_model)

        # initialize server
        self.server = server_classes[
            self.args.fed_algo]  # instantiate the server class corresponding to the selected federated algorithm
        self.client = client_classes[self.args.fed_algo](self.args, self.train_dataset, self.user_groups, self.logger,
                                                         self.server)  # instantiate the client class corresponding to the selected federated algorithm

        # initialize homomorphic encryption
        self.he = HEScheme(self.args.he_scheme_name)
        self.context = self.he.context
        self.secret_key = self.context.secret_key()  # save the secret key before making context public
        self.context.make_context_public()  # make the context object public so it can be shared across clients

    def train_clients(self, test=False, save=False, plot=False):
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
            print(f'\n | Global Training Round : {round_ + 1} |\n')

            self.global_model.train()

            # Call the client's train_clients method to perform local training and aggregation.
            train_loss, train_acc, global_weights, self.global_model_non_encry = self.client.train_clients(
                self.global_model, round_, self.global_model_non_encry)

            self.global_model.load_state_dict(copy.deepcopy(global_weights))

            train_accuracy_list.append(train_acc)
            train_loss_list.append(train_loss)

            if (round_ + 1) % print_every == 0:
                print(f' \nAvg Training Stats after {round_ + 1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss_list[-1]))}')
                print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy_list[-1]))

        # If specified, evaluate/save/plot the trained model
        if test:
            self.test_model(train_accuracy_list)

        if save:
            self.save_model(train_loss_list, train_accuracy_list)

        if plot:
            self.plot(train_loss_list, train_accuracy_list)

    def test_model(self, train_accuracy):
        # Test inference after completion of training
        test_loader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)
        test_acc, test_loss = self.client.evaluate_model(self.global_model, test_loader)

        print(f' \n Results after {self.args.rounds} global rounds of training on encrypted model:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        if self.args.train_without_encryption:
            # Test inference on non encrypted model after completion of training
            test_acc_nonencry, test_loss_nonencr = self.client.evaluate_model(self.global_model_non_encry,
                                                                              test_loader)

            print(f' \n Results after {self.args.rounds} global rounds of training on non-encrypted model:')
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc_nonencry))

    def save_model(self, train_loss, train_accuracy):
        # Saving the objects train_loss and train_accuracy:
        file_name = '../pretrained/{}_{}_iid[{}]_E[{}]_B[{}].pkl'. \
            format(self.args.dataset, self.args.rounds, self.args.iid,
                   self.args.epochs, self.args.local_bs)
        time.sleep(3)
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

    def plot(self, train_loss, train_accuracy):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('../reports/fed_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(self.args.dataset, self.args.rounds,
                           self.args.iid, self.args.epochs, self.args.local_bs))

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('../reports/fed_{}_{}_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(self.args.dataset, self.args.rounds,
                           self.args.iid, self.args.epochs, self.args.local_bs))
