import copy

import torch
from pympler import asizeof

from .FedClient import FedClient, measure_time


class GradientBasedFedAvgClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, json_logs, server):
        super().__init__(args, train_dataset, user_groups, json_logs, server)
        self.args = args
        self.print_every = 2

    @measure_time
    def aggregate_by_gradients(self, local_gradients, client_weights):
        # Get shapes of gradients
        weight_shapes = {k: v.shape for k, v in local_gradients[0].items()}
        self.json_logs["local_weights_size"] = asizeof.asizeof(local_gradients)

        # Encrypt client gradients
        encrypted_gradients = self.encrypt_weights(local_gradients)
        self.json_logs["encrypted_weights_size"] = asizeof.asizeof(local_gradients)

        # Aggregate encrypted gradients
        global_averaged_encrypted_gradients = self.aggregate_weights(
            encrypted_gradients
        )
        self.json_logs["global_averaged_encrypted_weights_size"] = asizeof.asizeof(
            global_averaged_encrypted_gradients
        )

        # Decrypt and average the gradients
        global_decrypted_averaged_gradients = self.decrypt_weights(
            global_averaged_encrypted_gradients, weight_shapes
        )
        self.json_logs["global_averaged_decr_weights_size"] = asizeof.asizeof(
            global_decrypted_averaged_gradients
        )

        for key, value in global_decrypted_averaged_gradients.items():
            global_decrypted_averaged_gradients[key] = value.to(self.device)

        return global_decrypted_averaged_gradients

    def get_grads_for_local_models(self, local_weights, global_model):
        """return a dict: {name: params}"""
        local_grads = []
        for local_weights_cli in local_weights:
            state_dict = {}
            for name, param0 in global_model.state_dict().items():
                param1 = local_weights_cli[name]
                state_dict[name] = param0.detach() - param1.detach()
            local_grads.append(state_dict)
        return local_grads

    def update_model_weights_with_aggregated_grads(self, aggregated_grads, model):
        weights = model.state_dict()

        for name, grad in aggregated_grads.items():
            if len(weights[name].size()) != 0:
                weights[name] -= grad

        return weights

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
                local_model, self.valid_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)
            self.update_local_training_logs(wandb, local_train_loss, accuracy, loss)

        local_grads = self.get_grads_for_local_models(local_weights, global_model)
        client_weights = self.get_client_weights()
        # Aggregate local model weights into global model weights
        aggregated_grads = self.aggregate_by_gradients(local_grads, client_weights)

        global_model_weights = self.update_model_weights_with_aggregated_grads(
            aggregated_grads, global_model
        )
        if self.args.train_without_encryption:
            aggregated_grads_nonencr = self.server.average_weights_nonencrypted(
                local_grads
            )
            global_model_weights_nonencr = (
                self.update_model_weights_with_aggregated_grads(
                    aggregated_grads_nonencr, global_model_non_encry
                )
            )
            global_model_non_encry.load_state_dict(global_model_weights_nonencr)

        # Compute average training loss and accuracy across all clients
        train_loss = sum(local_losses) / len(local_losses)
        train_accuracy = sum(eval_acc) / len(eval_acc)
        return (
            train_loss,
            train_accuracy,
            global_model_weights,
            global_model_non_encry,
        )
