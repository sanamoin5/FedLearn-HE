import torch
from torch import nn

torch.manual_seed(1)


class MNIST_2NN_MLP(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(MNIST_2NN_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x