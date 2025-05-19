import torch

from torch import nn


class PatchLoss(nn.Module):

    def __init__(self, delegate: nn.Module):
        super().__init__()

        self.delegate = delegate

    def forward(self, predictions, targets):
        _, _, h, w = predictions.shape
        mask = torch.zeros_like(predictions)

        mask[:, h // 2 + 1, w // 2 + 1] = 1

        masked_prediction = predictions * mask
        masked_target = targets * mask

        return self.delegate(masked_prediction, masked_target)
