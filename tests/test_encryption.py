import unittest
from collections import OrderedDict

import torch
from models.homomorphic_encryption import HEScheme


class TestCkksEncryption(unittest.TestCase):
    def setUp(self):
        self.he = HEScheme("ckks")
        self.context = self.he.context
        self.secret_key = (
            self.context.secret_key()
        )  # save the secret key before making context public
        self.context.make_context_public()

    def test_forward_pass(self):
        # Create a dummy input tensor
        od = OrderedDict()
        od["conv1.weight"] = torch.tensor(
            [[0.164, -0.148, -0.065], [0.164, -0.148, -0.065]]
        )
        local_weights = [od]  # Batch size 1, 1 channel, 28x28 image
        shapes = {"conv1.weight": torch.Size([2, 3])}
        # Perform a forward pass through the model
        encr_weights = self.he.encrypt_client_weights(local_weights)
        decr_weights = self.he.decrypt_and_average_weights_ckks_bfv(
            encr_weights[0], shapes, 1, self.secret_key
        )["conv1.weight"]

        self.assertListEqual(
            local_weights[0]["conv1.weight"].tolist(), decr_weights.tolist()
        )


if __name__ == "__main__":
    unittest.main()
