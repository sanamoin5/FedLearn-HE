import torch
from torch import nn
from torchvision.models import resnet

torch.manual_seed(1)


class MnistModel(nn.Module):
    def __init__(self) -> None:
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


cifar_cnn = resnet.ResNet(
    resnet.Bottleneck,
    [3, 4, 6, 3],
    num_classes=10,
    zero_init_residual=False,
    groups=1,
    width_per_group=64,
    replace_stride_with_dilation=None,
)

# class MnistModel(nn.Module):
#     def __init__(self):
#         super(MnistModel, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(784, 11),
#             nn.ReLU(),
#             nn.Linear(11, 10)
#             ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
