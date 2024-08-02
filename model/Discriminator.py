import torch.nn as nn
from functions import ReverseLayerF


class Discriminator(nn.Module):
    def __init__(self, num_domains):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(4096, 1024),
        self.fc2 = nn.Linear(1024, 1024),
        self.fc3 = nn.Linear(1024, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lambda_p):
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.model(x)
        return x
