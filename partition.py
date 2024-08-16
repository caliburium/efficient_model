import torch
import torch.nn as nn


class PartitionedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_partitions):
        super(PartitionedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions

        # Create partitions of the linear layer
        self.linears = nn.ModuleList(
            [nn.Linear(in_features // num_partitions, out_features // num_partitions) for _ in range(num_partitions)])

    def forward(self, x):
        # Split input into partitions
        x_split = torch.chunk(x, self.num_partitions, dim=1)

        # Apply each linear layer to its partition
        out_split = [self.linears[i](x_split[i]) for i in range(self.num_partitions)]

        # Concatenate the outputs
        out = torch.cat(out_split, dim=1)
        return out


# Usage
in_features = 8
out_features = 4
num_partitions = 2
x = torch.randn(1, in_features)
model = PartitionedLinear(in_features, out_features, num_partitions)
output = model(x)
print(output)
