from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
from torch.utils import data
from torch import nn, Tensor, optim


@dataclass
class TrainerHistoryEntry:
    train: dict[str, float] = field(default_factory=lambda: {})
    eval: dict[str, float] = field(default_factory=lambda: {})

    def as_postfix(self):
        return self.train | self.eval


@dataclass
class TrainerFeedback:
    history: list[TrainerHistoryEntry]


class TrainableModule(nn.Module):

    def get_params(self):
        return {}

    def configure_optimizers(self) -> optim.Optimizer:
        pass

    def configure_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        pass


class AdamOptimizedModule(TrainableModule):

    def __init__(
        self,
        net: nn.Module,
        lr: float,
        weight_decay=0,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__()

        self.lr = lr
        self.net = net
        self.weight_decay = weight_decay
        self.scheduler = scheduler

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def configure_scheduler(
        self, optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        if self.scheduler:
            return self.scheduler(optimizer)

        return None

    def get_params(self):
        wrapper_params = {
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler": str(self.scheduler),
        }

        net_params = (
            self.net.get_params()
            if hasattr(self.net, "get_params") and callable(self.net.get_params)
            else {}
        )

        return wrapper_params | net_params


class BaseTrainer(ABC):

    def fit(
        self,
        model: TrainableModule,
        train_dataloader: data.DataLoader,
        eval_dataloader: Optional[data.DataLoader] = None,
        test_dataloader: Optional[data.DataLoader] = None,
    ) -> TrainerFeedback:
        pass

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        pass

    def predict_labeled(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        pass

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        pass
