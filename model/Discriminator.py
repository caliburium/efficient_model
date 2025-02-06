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

class Discriminator224(nn.Module):
    def __init__(self, num_domains, input_size=256*6*6):
        super(Discriminator224, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_domains)
        )

    def forward(self, x, lambda_p):
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.discriminator(x)
        return x