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


class PULoss(nn.Module):
    def __init__(self, prior, loss_fn=None, gamma=1.0, beta=0.0, nnpu=True):
        """
        PU loss for binary classification.

        Args:
            prior (float): Class prior (P(Y=1)), must be in (0,1).
            loss_fn (callable): Loss function, should be non-increasing (default: sigmoid(-x)).
            gamma (float): Weight for non-negative risk correction.
            beta (float): Bias correction threshold for non-negative risk.
            nnpu (bool): Use non-negative PU learning if True, otherwise unbiased PU.
        """
        super().__init__()
        if not (0 < prior < 1):
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: torch.sigmoid(-x))
        self.nnpu = nnpu
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t):
        """
        Args:
            x (Tensor): Logits (N,).
            t (Tensor): Labels (N,), should contain only 1 (positive) and -1 (unlabeled).
        Returns:
            Tensor: PU loss (scalar).
        """
        t = t.view(-1)
        x = x.view(-1)

        positive = (t == self.positive).float()
        unlabeled = (t == self.unlabeled).float()

        n_positive = max(positive.sum().item(), 1.0)
        n_unlabeled = max(unlabeled.sum().item(), 1.0)

        # PU loss components
        y_positive = self.loss_fn(x)
        y_unlabeled = self.loss_fn(-x)

        positive_risk = self.prior * (positive * y_positive).sum() / n_positive
        negative_risk = (
            (unlabeled * y_unlabeled) / n_unlabeled
            - self.prior * (positive * y_unlabeled) / n_positive
        ).sum()

        objective = positive_risk + negative_risk

        if self.nnpu:
            if negative_risk.item() < -self.beta:
                loss = positive_risk - self.beta
                # gradient of negative risk is scaled when it’s too negative
                loss += -self.gamma * negative_risk
            else:
                loss = objective
        else:
            loss = objective

        return loss
