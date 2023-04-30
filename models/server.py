import copy
from abc import ABC, abstractmethod
import torch


class Server(ABC):
    # perform homomorphic aggregation
    @abstractmethod
    def aggregate(self, encryption_type):
        pass

    def deserialize_data(self):
        pass

    def serialize_data(self):
        pass

    def average_weights_nonencrypted(w):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg


class FedAvgServer(Server):
    def __init__(self, encrypted_weights, client_weights=None):
        self.encrypted_weights = encrypted_weights
        if client_weights == None:
            self.client_weights = [1] * len(encrypted_weights)
        else:
            self.client_weights = client_weights

    def aggregate_ckks_bfv(self):
        w_sum = {}
        for encrypted_client_weights, client_weight in zip(self.encrypted_weights, self.client_weights):
            for key in encrypted_client_weights:
                if key not in w_sum:
                    w_sum[key] = 0
                w_sum[key] = w_sum[key] + encrypted_client_weights[key] * client_weight
        return w_sum

    def aggregate_paillier(self):
        w_sum = {}
        len_clients = len(self.encrypted_weights)
        param_key_list = list(self.encrypted_weights[0].keys())

        for key in param_key_list:
            list_param_values = []
            for cli_idx in range(len_clients):
                list_param_values.append(self.encrypted_weights[cli_idx][key])
            if key not in w_sum:
                w_sum[key] = 0
            temp_param_values_list = []
            param_value_range = len(list_param_values[0])
            for j in range(param_value_range):
                sum_list_item_values = 0
                for k in range(len_clients):
                    sum_list_item_values = sum_list_item_values + list_param_values[k][j] * self.client_weights[k]
                temp_param_values_list.append(sum_list_item_values)
            w_sum[key] = temp_param_values_list
        return w_sum

    def aggregate(self, encryption_type):
        w_sum = {}

        if encryption_type == 'ckks' or encryption_type == 'bfv':
            w_sum = self.aggregate_ckks_bfv()

        elif encryption_type == 'paillier':
            w_sum = self.aggregate_paillier()

        return w_sum
