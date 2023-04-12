import copy
from abc import ABC, abstractmethod
import torch


class Server(ABC):
    # perform homomorphic summation
    @abstractmethod
    def aggregate(self):
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
    def __init__(self, encrypted_weights):
        self.encrypted_weights = encrypted_weights

    def aggregate(self):
        w_sum = {}
        for encrypted_client_weights in self.encrypted_weights:
            for key in encrypted_client_weights:
                if key not in w_sum:
                    w_sum[key] = 0
                w_sum[key] += encrypted_client_weights[key]

        return w_sum


class WeightedFedAvgServer(Server):
    def __init__(self, encrypted_weights_client_weights):
        self.encrypted_weights = encrypted_weights_client_weights[0]
        self.client_weights = encrypted_weights_client_weights[1]

    def aggregate(self):
        w_sum = {}
        for encrypted_client_weights, client_weight in zip(self.encrypted_weights, self.client_weights):
            for key in encrypted_client_weights:
                if key not in w_sum:
                    w_sum[key] = 0
                w_sum[key] += encrypted_client_weights[key] * client_weight

        return w_sum
