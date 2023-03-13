import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from args_parser import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, encrypted_weights_sum
from nn_models import ResNetModel, MnistModel
from homomorphic_encryption import HEScheme
from server import Server


class Client:
    def __init__(self):
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        # print details of the experiment
        exp_details(self.args)

        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

        # BUILD MODEL
        if self.args.dataset == 'mnist':
            self.global_model = MnistModel()
            MnistModel()
        elif self.args.dataset == 'cifar':
            self.global_model = ResNetModel()

        else:
            exit('Error: unrecognized model')

        if self.args.test_without_encryption:
            self.global_model_non_encry = copy.deepcopy(self.global_model)

        # initialize homomorphic encryption
        self.he = HEScheme(self.args.he_scheme_name)
        self.context = self.he.context
        self.secret_key = self.context.secret_key()
        self.context.make_context_public()

        # initialize server
        self.server = Server(self.args.fed_algo, self.context)

    def aggregate_by_weights(self, local_weights):

        # update global weights without encryption
        global_weights = average_weights(local_weights)
        if self.args.test_without_encryption:
            self.global_model_non_encry.load_state_dict(global_weights)
        # update global weights with encryption
        shapes = {}
        for key, value in local_weights[0].items():
            shapes[key] = value.shape

        encrypted_weights = self.he.encrypt_client_weights(local_weights)
        # TODO: make use of context in averaging after serializing input and context
        # using server class method to encrypt weights
        # global_averaged_encrypted_weights = server.encrypted_weights_sum(encrypted_weights)

        # using local function to encrypt weights
        global_averaged_encrypted_weights = encrypted_weights_sum(encrypted_weights)
        global_weights = self.he.decrypt_and_average_weights(global_averaged_encrypted_weights, shapes,
                                                             self.args.num_users,
                                                             self.secret_key)

        # update global weights
        self.global_model.load_state_dict(global_weights)

    def aggregate_by_gradients(self, temp_model):
        bound = 2 ** 3
        prec = 32
        params_modules = list(temp_model.named_parameters())
        params_grad_list = []
        for params_module in params_modules:
            name, params = params_module
            params_grad_list.append(copy.deepcopy(params.grad).view(-1))

        params_grad = ((torch.cat(params_grad_list, 0) + bound) * 2 ** prec).long()

    def train_model(self, test=False, save=False, plot=False):
        # Set the model to train and send it to device.
        self.global_model.to(self.device)
        self.global_model.train()

        if self.args.test_without_encryption:
            self.global_model_non_encry.to(self.device)
            self.global_model_non_encry.train()

        print(self.global_model)

        # copy weights
        self.global_weights = self.global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        print_every = 2

        for round_ in tqdm(range(self.args.rounds)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {round_ + 1} |\n')

            self.global_model.train()
            m = self.args.num_users

            idxs_users = range(m)

            for idx in idxs_users:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)
                w, loss, temp_model = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=round_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            self.aggregate_by_gradients(temp_model)
            self.aggregate_by_weights(local_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = self.evaluate_model(idx)

            train_accuracy.append(sum(list_acc) / len(list_acc))

            # print global training loss after every 'i' rounds
            if (round_ + 1) % print_every == 0:
                print(f' \nAvg Training Stats after {round_ + 1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if test:
            self.test_model(train_accuracy)
        if save:
            self.save_model(train_loss, train_accuracy)
        if plot:
            self.plot(train_loss, train_accuracy)

    def evaluate_model(self, idx):
        list_acc, list_loss = [], []
        self.global_model.eval()
        for c in range(self.args.num_users):
            local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                      idxs=self.user_groups[idx], logger=self.logger)
            acc, loss = local_model.inference(model=self.global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        return list_acc, list_loss

    def test_model(self, train_accuracy):
        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)

        print(f' \n Results after {self.args.rounds} global rounds of training on encrypted model:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        if self.args.test_without_encryption:
            # Test inference on non encrypted model after completion of training
            test_acc_nonencry, test_loss_nonencr = test_inference(self.args, self.global_model_non_encry,
                                                                  self.test_dataset)

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

    @staticmethod
    def serialize_encrypted_data():
        pass

    @staticmethod
    def deserialize_encrypted_data():
        pass

    def convert_float_to_int(self):
        pass

    def convert_int_to_float(self):
        pass

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
