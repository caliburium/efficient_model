import torch


def coral_loss(source, target):
    d = source.size(1)

    source_covar = torch.mm(source.T, source) / source.size(0)
    target_covar = torch.mm(target.T, target) / target.size(0)

    loss = torch.mean(torch.square(source_covar - target_covar))
    loss = loss / (4 * d * d)

    return loss
