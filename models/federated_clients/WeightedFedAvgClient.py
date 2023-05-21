import copy

from .FedClient import FedClient


class WeightedFedAvgClient(FedClient):
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

    def aggregate_by_weights(self, local_weights, client_weights):
        # Get shapes of weights
        weight_shapes = {k: v.shape for k, v in local_weights[0].items()}

        # Encrypt client weights
        encrypted_weights = self.he.encrypt_client_weights(local_weights)

        # Aggregate encrypted weights
        global_averaged_encrypted_weights = self.server(
            encrypted_weights, client_weights
        ).aggregate(self.args.he_scheme_name)

        # Decrypt and average the weights
        global_averaged_weights = self.he.decrypt_and_average_weights(
            global_averaged_encrypted_weights,
            weight_shapes,
            client_weights=sum(client_weights),
            secret_key=self.secret_key,
        )
        return global_averaged_weights

    def train_clients(self, global_model, round_, global_model_non_encry=None):
        local_weights, local_losses, eval_acc, eval_losses = [], [], [], []

        # Iterate over each client
        for client_idx in range(self.args.num_clients):
            # Train local model
            local_model, loss = self.train_local_model(
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
