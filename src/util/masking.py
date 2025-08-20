from abc import ABC

import torch

from src.util.hsi import mask_hsi_batch


class Maksking(ABC):
    def mask(x):
        pass


class HsiPatchMasking(Maksking):

    def __init__(self, mask_ratio=0.75, mode="spatial", fill_value=0.0):
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.fill_value = fill_value

    def mask(self, batch):
        return mask_hsi_batch(
            batch,
            mask_ratio=self.mask_ratio,
            mode=self.mode,
            fill_value=self.fill_value,
        )


class RandomFlipMaskingDecorator(Maksking):

    def __init__(
        self,
        decorated: Maksking,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
    ):
        self.decorated = decorated
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def mask(self, batch):
        batch = torch.flip(batch, dims=[-1])
        batch = torch.flip(batch, dims=[-2])

        return mask_hsi_batch(
            batch,
            mask_ratio=self.mask_ratio,
            mode=self.mode,
            fill_value=self.fill_value,
        )
