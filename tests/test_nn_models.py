import unittest

import torch
from models.nn_models import MnistModel


class TestMnistModel(unittest.TestCase):
    def setUp(self):
        self.model = MnistModel()

    def test_forward_pass(self):
        # Create a dummy input tensor
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image

        # Perform a forward pass through the model
        output = self.model(input_tensor)

        # Ensure the output tensor has the expected shape
        expected_shape = (1, 12)  # Batch size 1, 10 classes
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
