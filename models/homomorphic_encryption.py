import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F

import tenseal as ts
from phe import paillier
import collections
import time


# use <T> kind to initialize he
class HEScheme:
    def __init__(self, he_scheme_name, bits_scale=40, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=[40, 40, 40, 40, 40], global_scale=40, create_galois_keys=False,
                 plain_modulus=1032193):
        self.bits_scale = bits_scale
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.plain_modulus = plain_modulus
        self.create_galois_keys = create_galois_keys

        if he_scheme_name == 'ckks':
            self.init_ckks()
        elif he_scheme_name == 'paillier':
            self.init_paillier()
        elif he_scheme_name == 'bfv':
            self.init_bfv()

    def init_ckks(self):
        # controls precision of the fractional part
        bits_scale = self.bits_scale

        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            # coeff_mod_bit_sizes=[60, bits_scale, bits_scale, 60]
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            # coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
            # coeff_mod_bit_sizes = [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        )
        self.context.global_scale = 2 ** self.global_scale
        if self.create_galois_keys:
            self.context.generate_galois_keys()
        self.encrypt_function = ts.ckks_vector
        # self.decrypt_func =
        # pack all channels into a single flattened vector
        # enc_x = ts.CKKSVector.pack_vectors(enc_channels)

    def init_paillier(self):
        public_key, private_key = paillier.generate_paillier_keypair()
        secret_number_list = [3.141592653, 300, -4.6e-12]

        # secret_number_list = [3.141592653, 300, -4.6e-12]
        # encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]
        # [private_key.decrypt(x) for x in encrypted_number_list]
        # a_times_3_5_lp = a * paillier.EncodedNumber.encode(a.public_key, 3.5, 1e-2) # reducing precision

    def init_bfv(self):
        self.context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
        self.encrypt_function = ts.bfv_vector
        # encrypted_vector = ts.bfv_vector(context, plain_vector)

    def serialize_encrypted_data(self):
        pass

    def deserialize_encrypted_data(self):
        pass

    def encrypt_client_weights(self, clients_weights, secret_key=None) -> list:
        encr = []
        for client_weights in clients_weights:
            encr_state_dict = {}
            for key, value in client_weights.items():
                val = value.flatten()
                encr_state_dict[key] = self.encrypt_function(self.context, val)
            encr.append(encr_state_dict)
        return encr

    def decrypt_and_average_weights(self, encr_weights, shapes, client_weights, secret_key=None):
        decry_model = {}
        for key, value in encr_weights.items():
            decry_model[key] = torch.reshape(torch.tensor(value.decrypt(secret_key)), shapes[key])
            # average weights
            decry_model[key] = torch.div(decry_model[key], client_weights)
        return decry_model


def get_client_encrypted_grad(client_inputs, client_labels, net_dict, client_net):
    client_net.load_state_dict(net_dict)
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.SGD(client_net.parameters(), lr=LR, momentum=0.9)
    client_optimizer.zero_grad()  # 梯度置零
    client_loss.backward()  # 求取梯度

    params_modules = list(client_net.named_parameters())
    params_grad_list = []
    for params_module in params_modules:
        name, params = params_module
        params_grad_list.append(copy.deepcopy(params.grad).view(-1))

    params_grad = ((torch.cat(params_grad_list, 0) + bound) * 2 ** prec).long().cuda()
    client_encrypted_grad = Enc(pk, params_grad)

    client_optimizer.zero_grad()

    return client_encrypted_grad
