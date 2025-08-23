from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
from torch.utils import data
from torch import nn, Tensor, optim

from src.util.scheduler import CosineLRSchedulerWrapper


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

    def get_display_name(self) -> Optional[str]:
        pass

    def get_params(self) -> dict[str, object]:
        return {}

    def configure_optimizers(self) -> optim.Optimizer:
        pass

    def configure_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        pass


class SchedulerProvider(ABC):

    def provide(self, optimizer: optim.Optimizer):
        pass

    def get_params(self):
        return {}


class NoneSchedulerProvider(SchedulerProvider):

    def provide(self, optimizer: optim.Optimizer):
        return None


class LrSchedulerProvider(SchedulerProvider):

    def __init__(self, t_initial, lr_min, warmup_t, warmup_lr_init):
        super().__init__()

        self.params = {
            "schduler_type": "CosineLRSchedulerWrapper",
            "t_initial": t_initial,
            "lr_min": lr_min,
            "warmup_t": warmup_t,
            "warmup_lr_init": warmup_lr_init,
        }

        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init

    def provide(self, optimizer):
        return CosineLRSchedulerWrapper(
            optimizer,
            t_initial=self.t_initial,
            lr_min=self.lr_min,
            warmup_t=self.warmup_t,
            warmup_lr_init=self.warmup_lr_init,
        )

    def get_params(self):
        return self.params


class AdamOptimizedModule(TrainableModule):

    def __init__(
        self,
        net: nn.Module,
        lr: float,
        weight_decay=0,
        scheduler: SchedulerProvider = NoneSchedulerProvider(),
        display_name: Optional[str] = None,
        no_decay: Optional[list[str]] = None,
    ):
        super().__init__()

        self.lr = lr
        self.net = net
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.display_name = display_name
        self.no_decay = no_decay

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)

    def configure_optimizers(self) -> optim.Optimizer:
        if not self.no_decay:
            return optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in self.no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in self.no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            return optim.AdamW(
                optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
            )

    def configure_scheduler(
        self, optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        return self.scheduler.provide(optimizer)

    def get_display_name(self):
        return self.display_name

    def get_params(self) -> dict[str, object]:
        wrapper_params = {
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler.get_params(),
        }

        net_params = (
            self.net.get_params()
            if hasattr(self.net, "get_params") and callable(self.net.get_params)
            else {}
        )

        return wrapper_params | net_params


class BaseTrainer(ABC):

    def get_params(self):
        pass

    def fit(
        self,
        model: TrainableModule,
        train_dataloader: data.DataLoader,
        eval_dataloader: Optional[data.DataLoader] = None,
        test_dataloader: Optional[data.DataLoader] = None,
    ) -> tuple[TrainerFeedback, TrainableModule]:
        pass

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        pass

    def predict_labeled(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        model.to(self.device)
        model.eval()

        all_x = []
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = model(x)

                all_x.append(x)
                all_y_true.append(y)
                all_y_pred.append(y_pred)

        return all_x, all_y_true, all_y_pred

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        pass
