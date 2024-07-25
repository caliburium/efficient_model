import torch.nn as nn
from functions import ReverseLayerF


class CaffeNet(nn.Module):
    def __init__(self):
        super(CaffeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm = nn.LocalResponseNorm(5, 1.e-4, 0.75)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.norm(self.pool(self.relu(self.conv1(x * 57.6))))
        x = self.norm(self.pool(self.relu(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        return x


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_domains=4):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lambda_p):
        x = x.view(-1, 256 * 6 * 6)
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),  # 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),  # 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),  # 8
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_data, alpha=1.0, discriminator=1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)
        class_output = self.classifier(feature)
        if discriminator == 1:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.discriminator(reverse_feature)
        elif discriminator == 0:
            domain_output = 0
        return class_output, domain_output