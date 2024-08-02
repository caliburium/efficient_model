import torch.nn as nn
from functions import ReverseLayerF


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),  # 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),  # 1
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, input_data, alpha=1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output
