import copy

import torch
from pympler import asizeof

from .FedClient import FedClient, measure_time, WeightedFedAvgClient



class MultiModalFedClient(WeightedFedAvgClient):
    def __init__(self, args, train_dataset, user_groups, json_logs, server):
        super().__init__(args, train_dataset, user_groups, json_logs, server)
        self.args = args
        self.print_every = 2

    def aggregate_by_weights(self, local_weights, client_weights):
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}
        self.json_logs["local_weights_size"] = asizeof.asizeof(local_weights)

        # Encrypt client weights
        encrypted_weights = self.encrypt_weights(local_weights)
        self.json_logs["encrypted_weights_size"] = asizeof.asizeof(encrypted_weights)

        # Aggregate encrypted weights
        global_averaged_encrypted_weights = self.aggregate_weights(encrypted_weights)
        self.json_logs["global_averaged_encrypted_weights_size"] = asizeof.asizeof(
            global_averaged_encrypted_weights
        )

        # Decrypt and average the weights
        global_averaged_weights = self.decrypt_weights(
            global_averaged_encrypted_weights, weight_shapes
        )
        self.json_logs["global_averaged_decr_weights_size"] = asizeof.asizeof(
            global_averaged_weights
        )

        return global_averaged_weights

    @measure_time
    def train_clients(self, global_model, round_, wandb, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, local_train_loss = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(local_train_loss))
            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)
            self.update_local_training_logs(wandb, local_train_loss, accuracy, loss)

        client_weights = self.get_client_weights()
        # Aggregate local model weights into global model weights
        aggregated_weights = self.aggregate_by_weights(local_weights, client_weights)

        if self.args.train_without_encryption:
            global_weights = self.server.average_weights_nonencrypted(local_weights)
            global_model_non_encry.load_state_dict(global_weights)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)

        return train_loss, train_accuracy, aggregated_weights, global_model_non_encry
