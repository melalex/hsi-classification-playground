from abc import ABC
from dataclasses import dataclass, field
from typing import Optional
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

    def configure_optimizers(self) -> optim.Optimizer:
        pass

    def configure_scheduller(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        pass


class AdamOptimizedModule(TrainableModule):

    def __init__(
        self,
        net: nn.Module,
        lr: float,
        weight_decay=0,
        scheduler_step_size: Optional[int] = None,
        scheduler_gamma: Optional[float] = None,
    ):
        super().__init__()

        self.lr = lr
        self.net = net
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def configure_scheduller(
        self, optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        if self.scheduler_step_size and self.scheduler_gamma:
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )

        return None


class BaseTrainer(ABC):

    def fit(
        self,
        model: TrainableModule,
        train_dataloader: data.DataLoader,
        eval_dataloader: Optional[data.DataLoader],
    ) -> TrainerFeedback:
        pass

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        pass

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        pass
