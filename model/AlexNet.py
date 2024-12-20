import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functions.ReverseLayerF import ReverseLayerF

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNetDANN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetDANN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
        )

    # DEFINE HOW FORWARD PASS IS COMPUTED
    def forward(self, x, lambda_p):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        reversed_feature = ReverseLayerF.apply(features, lambda_p)
        label_out = self.classifier(features)
        domain_out = self.discriminator(reversed_feature)

        return label_out, domain_out


def DANN_Alex(pretrained=True, progress=True, **kwargs):
    model = AlexNetDANN(num_classes=1000, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    # Change output classes
    model.classifier[6] = nn.Linear(4096, 7)
    model.discriminator[6] = nn.Linear(4096, 4)

    # Copy pretrained weights from the classifier to the discriminator
    model.discriminator[1].weight.data = model.classifier[1].weight.data.clone()
    model.discriminator[1].bias.data = model.classifier[1].bias.data.clone()

    model.discriminator[4].weight.data = model.classifier[4].weight.data.clone()
    model.discriminator[4].bias.data = model.classifier[4].bias.data.clone()

    return model


class AlexNetCaffe(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetCaffe, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # DEFINE HOW FORWARD PASS IS COMPUTED
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        label_out = self.classifier(features)

        return label_out


def AlexNet(pretrained=True, progress=True, **kwargs):
    model = AlexNetCaffe(num_classes=1000, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    # Change output classes
    model.classifier[6] = nn.Linear(4096, 7)

    return model
