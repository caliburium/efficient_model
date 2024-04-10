# https://github.com/fungtion/DANN/blob/master/models/functions.py
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None


def lr_lambda(progress):
    mu_0 = 0.01
    alpha = 10
    beta = 0.75
    return mu_0 * (1 + alpha * progress) ** (-beta)