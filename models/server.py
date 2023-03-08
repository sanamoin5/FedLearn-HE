import copy

class Server:
    def __init__(self, aggregation_scheme_name='FedAvg', context=None):
        self.aggregation_scheme_name = aggregation_scheme_name
        self.context = context

    # perform homomorphic summation based on aggregation scheme
    def encrypted_weights_sum(self, encrypted_weights):
        """
        Returns the average of the weights with .
        """
        if self.aggregation_scheme_name == 'FedAvg':
            return self.fedavg_aggregation(encrypted_weights)

    def fedavg_aggregation(self, encrypted_weights):
        w_sum = copy.deepcopy(encrypted_weights[0])
        for key in w_sum.keys():
            for i in range(1, len(encrypted_weights)):
                w_sum[key] += encrypted_weights[i][key]
        return w_sum

    def deserialize_inputs(self):
        pass

