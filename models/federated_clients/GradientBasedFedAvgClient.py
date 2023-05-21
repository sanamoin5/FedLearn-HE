import copy

from .FedClient import FedClient


class GradientBasedFedAvgClient(FedClient):
    def __init__(self, args, train_dataset, user_groups, logger, server):
        super().__init__(args, train_dataset, user_groups, logger, server)
        self.args = args
        self.print_every = 2

        # Initialize train, validation and test loaders for each client
        self.train_loader, self.valid_loader, self.test_loader = [], [], []
        for idx in range(self.args.num_clients):
            train_loader, valid_loader, test_loader = self.train_val_test(
                list(self.user_groups[idx])
            )
            self.train_loader.append(train_loader)
            self.valid_loader.append(valid_loader)
            self.test_loader.append(test_loader)

    def aggregate_by_gradients(self, local_gradients, client_weights):
        # Get shapes of gradients
        weight_shapes = {k: v.shape for k, v in local_gradients[0].items()}

        # Encrypt client gradients
        encrypted_gradients = self.he.encrypt_client_weights(local_gradients)

        # Aggregate encrypted gradients
        global_averaged_encrypted_gradients = self.server(
            encrypted_gradients, client_weights
        ).aggregate(self.args.he_scheme_name)

        # Decrypt and average the gradients
        global_decrypted_averaged_gradients = self.he.decrypt_and_average_weights(
            global_averaged_encrypted_gradients,
            weight_shapes,
            client_weights=sum(client_weights),
            secret_key=self.secret_key,
        )
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
            weights[name] -= grad

        return weights

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss, time_training = self.train_local_model(
                model=copy.deepcopy(global_model),
                global_round=round_,
                client_idx=client_idx,
            )

            # Collect local model weights and losses
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(copy.deepcopy(loss))

            # Evaluate local model on test dataset
            accuracy, loss = self.evaluate_model(
                local_model, self.test_loader[client_idx]
            )
            eval_acc.append(accuracy)
            eval_losses.append(loss)

        local_grads = self.get_grads_for_local_models(local_weights, global_model)
        client_weights = self.get_client_weights()
        # Aggregate local model weights into global model weights
        aggregated_grads, time_processing = self.aggregate_by_gradients(
            local_grads, client_weights
        )

        global_model_weights = self.update_model_weights_with_aggregated_grads(
            aggregated_grads, global_model
        )
        time_processing["time_training_local"] = time_training
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
            time_processing,
        )
