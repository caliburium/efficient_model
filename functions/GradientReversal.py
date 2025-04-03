import torch

class GradientReversal(torch.nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        x.register_hook(lambda grad: -self.lambd * grad)
        return x
