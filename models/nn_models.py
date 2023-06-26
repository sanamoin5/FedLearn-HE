import torch
from torch import nn
from torchvision.models import resnet
from models.utils import BackboneBuilder
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
    [1,1,1,1],
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
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
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

        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)
        # feature fusion
        result = self.classifier(fused_data)

        return result, attention