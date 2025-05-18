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


class AdamOptimizedModule(TrainableModule):

    def __init__(self, net: nn.Module, lr: float):
        super().__init__()

        self.lr = lr
        self.net = net

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


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
