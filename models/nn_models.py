import torch
from torch import nn
from torchvision.models import resnet
from models.utils import BackboneBuilder
import torch.nn.functional as F

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
#python main.py --epochs 10 --round 500  --fed_algo FedAvg --he_scheme_name ckks  --gpu_id 1 --model mnist_2nn --optimizer sgd --num_clients 10 --lr 0.088 --local_bs 20

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class CIFAR10_CNN(nn.Module):
    """from torch tutorial
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


CIFAR10_ResNet = resnet.ResNet(
    resnet.Bottleneck,
    [1, 1, 1, 1],
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

class AttentionAggregator(nn.Module):
    """Aggregate features with computed attention value."""

    def __init__(self, in_features_size, inner_feature_size=128, out_feature_size=256):
        super(AttentionAggregator, self).__init__()

        self.in_features_size = in_features_size  # size of flatten feature
        self.L = out_feature_size
        self.D = inner_feature_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.in_features_size, self.L),
            nn.Dropout(),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(self.D, 1),
            nn.Dropout()
        )

    def forward(self, x):
        x = x.view(-1, self.in_features_size)  # flatten feature，[N, C * H * W]
        x = self.fc1(x)  # [B, L]

        a = self.attention(x)  # attention value，[N, 1]
        a = torch.transpose(a, 1, 0)  # [1, N]
        a = torch.softmax(a, dim=1)

        m = torch.mm(a, x)  # [1, N] * [N, L] = [1, L]

        return m, a


class MILNetWithClinicalData(nn.Module):
    """Training with image and clinical data"""

    def __init__(self, num_classes, backbone_name, clinical_data_size=5, expand_times=10) -> None:
        super(MILNetWithClinicalData, self).__init__()

        print('training with image and clinical data')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size,
                                                        1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L + self.clinical_data_size * self.expand_times, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        return clinical_data, aggregated_feature, attention

        # fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)
        # # feature fusion
        # result = self.classifier(fused_data)
        #
        # return result, attention
