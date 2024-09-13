import torch.nn as nn
from functions.ReverseLayerF import ReverseLayerF


class Discriminator32(nn.Module):
    def __init__(self, num_domains):
        super(Discriminator32, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 8 * 8, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_domains)
        )

    def forward(self, x, lambda_p):
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.discriminator(x)
        return x

class Discriminator227(nn.Module):
    def __init__(self, num_domains):
        super(Discriminator227, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_domains)
        )

    def forward(self, x, lambda_p):
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.discriminator(x)
        return x