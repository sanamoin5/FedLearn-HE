#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


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

def client():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL

    if args.dataset == 'mnist':
        global_model = MnistModel()
        global_model_non_encry = MnistModel()
    elif args.dataset == 'cifar':
        global_model = ResNetModel()
        global_model_non_encry = ResNetModel()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    global_model_non_encry.to(device)
    global_model_non_encry.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # initialize homomorphic encryption
    he = HEScheme('ckks')
    context = he.context
    secret_key = context.secret_key()
    context.make_context_public()

    # initialize server
    serv = Server('FedAvg', context)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for round_ in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round_+1} |\n')

        global_model.train()
        m = args.num_users
        # m = max(int(args.frac * args.num_users), 1)
        idxs_users = range(m)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=round_)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights without encryption
        global_weights = average_weights(local_weights)
        global_model_non_encry.load_state_dict(global_weights)
        # update global weights with encryption
        shapes = {}
        for key, value in local_weights[0].items():
            shapes[key] = value.shape

        encrypted_weights = he.encrypt_client_weights(local_weights)
        # TODO: make use of context in averaging after serializing input and context
        # using server class method to encrypt weights
        # global_averaged_encrypted_weights = serv.encrypted_weights_sum(encrypted_weights)

        # using local function to encrypt weights
        global_averaged_encrypted_weights = encrypted_weights_sum(encrypted_weights)
        global_weights = he.decrypt_and_average_weights(global_averaged_encrypted_weights, shapes, args.num_users, secret_key)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (round_+1) % print_every == 0:
            print(f' \nAvg Training Stats after {round_+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.rounds} global rounds of training on encrypted model:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))



    # Test inference after completion of training
    test_acc_nonencry, test_loss_nonencr = test_inference(args, global_model_non_encry, test_dataset)

    print(f' \n Results after {args.rounds} global rounds of training on non-encrypted model:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc_nonencry))


    # Saving the objects train_loss and train_accuracy:
    file_name = '../pretrained/{}_{}_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset,  args.rounds, args.iid,
               args.epochs, args.local_bs)
    time.sleep(3)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.rounds,
    #                    args.iid, args.epochs, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.rounds,
    #                    args.iid, args.epochs, args.local_bs))




