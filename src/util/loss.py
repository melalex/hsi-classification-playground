import torch

import torch.nn.functional as F

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


class PuLoss(nn.Module):

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


class MaskedAutoencoderLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=0.1, use_huber=False, huber_delta=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.use_huber = use_huber
        self.huber_delta = huber_delta

    def forward(self, recon, target, mask):
        loss, _ = self.__combined_loss(recon, target, mask)

        return loss

    def __masked_mse_loss(self, recon, target, mask):
        """
        recon, target, mask: [B, C, H, W]
        mask: 1 where masked, 0 where unmasked (can also be boolean)
        """
        masked = mask.to(recon.dtype)
        diff = (recon - target) * masked
        mse = (diff**2).sum() / (masked.sum() + 1e-12)
        return mse

    def __masked_cosine_spectral_loss(self, recon, target, mask):
        """
        Cosine spectral loss computed per pixel spectrum.
        recon, target: [B, C, H, W]
        mask: [B, C, H, W] — 1 for masked bands, 0 for unmasked
        """
        B, C, H, W = recon.shape

        # Flatten spatial dims
        recon_flat = recon.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)
        mask_flat = mask.permute(0, 2, 3, 1).reshape(-1, C)

        # Keep only masked spectral bands
        valid_mask = mask_flat.sum(dim=1) > 0  # at least one masked band in pixel
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=recon.device)

        r = recon_flat[valid_mask]
        t = target_flat[valid_mask]
        m = mask_flat[valid_mask]

        # Zero-out unmasked bands before cosine computation
        r = r * m
        t = t * m

        # Normalize
        r_norm = F.normalize(r, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)

        cos_sim = (r_norm * t_norm).sum(dim=-1)  # in [-1,1]
        return (1.0 - cos_sim).mean()

    def __combined_loss(self, recon, target, mask):
        """
        recon, target, mask: [B, C, H, W]
        mask: per-band mask
        """
        if self.use_huber:
            valid = mask > 0
            if valid.sum() == 0:
                rec_loss = torch.tensor(0.0, device=recon.device)
            else:
                rec_loss = F.smooth_l1_loss(
                    recon[valid], target[valid], reduction="mean", beta=self.huber_delta
                )
        else:
            rec_loss = self.__masked_mse_loss(recon, target, mask)

        spec_loss = self.__masked_cosine_spectral_loss(recon, target, mask)
        return self.alpha * rec_loss + self.beta * spec_loss, {
            "mse": rec_loss.item(),
            "spec": spec_loss.item(),
        }


class MaskedAutoencoderWithClassificationHeadLoss(nn.Module):

    def __init__(
        self,
        autoencoder_loss: nn.Module,
        cls_loss: nn.Module,
        cls_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.autoencoder_loss = autoencoder_loss
        self.cls_loss = cls_loss
        self.cls_loss_weight = cls_loss_weight

    def forward(self, recon, x, mask, y_pred, y_true):
        autoencoder_loss = self.autoencoder_loss(recon, x, mask)
        cls_loss = self.cls_loss(y_pred, y_true)

        return (
            autoencoder_loss,
            cls_loss,
            (autoencoder_loss + self.cls_loss_weight * cls_loss),
        )


class CompositeBinClassificationLoss(nn.Module):

    def __init__(self, loss_by_class: dict[int, nn.Module]):
        super().__init__()
        self.loss_by_class = nn.ModuleDict(
            {str(k): v for k, v in loss_by_class.items()}
        )

    def forward(self, y_pred, y_true):
        """
        y_pred: [B, C] logits for each class
        y_true: [B] true class indices (1..C)
        """
        total_loss = 0.0
        for class_id, loss_fn in self.loss_by_class.items():
            class_id = int(class_id)
            class_idx = (
                class_id - 1
            )  # since your dict starts at 1, but preds are 0-based

            # predictions for this class
            pred = y_pred[:, class_idx]

            # binary labels: 1 if this sample belongs to this class, else -1/0 depending on your loss
            target = torch.where(
                y_true == class_id,
                torch.ones_like(y_true, dtype=torch.float),
                -torch.ones_like(y_true, dtype=torch.float),
            )

            # call the loss function
            class_loss = loss_fn(pred, target)

            total_loss += class_loss

        return total_loss


def zero_one_loss(z):
    # (1 - sign(z)) / 2
    return (1 - torch.sign(z)) / 2


def ramp_loss(z):
    # max(0, min(1, (1 - z) / 2))
    return torch.clamp((1 - z) / 2, min=0, max=1)


def squared_loss(z):
    # (z - 1)^2 / 4
    return ((z - 1) ** 2) / 4


def logistic_loss(z):
    # ln(1 + exp(-z))
    return F.softplus(-z)


def hinge_loss(z):
    # max(0, 1 - z)
    return torch.clamp(1 - z, min=0)


def double_hinge_loss(z):
    # max(0, (1 - z) / 2, -z)
    return torch.max(torch.zeros_like(z), torch.max((1 - z) / 2, -z))


def sigmoid_loss(z):
    # 1 / (1 + exp(z))
    return torch.sigmoid(-z)
