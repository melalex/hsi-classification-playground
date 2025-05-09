from typing import Optional
import numpy as np

from dataclasses import dataclass
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader


@dataclass
class AutoEncoderMetrics:
    loss: float


@dataclass
class AutoEncoderTrainerHistoryEntry:
    train: AutoEncoderMetrics
    eval: Optional[AutoEncoderMetrics]


@dataclass
class TrainFeedBack:
    history: list[AutoEncoderTrainerHistoryEntry]


class AutoEncoderTrainer:

    def __init__(self, model, loss_fun, epochs, optimizer):
        self.model = model
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(
        self, train: DataLoader, eval: Optional[DataLoader] = None
    ) -> TrainFeedBack:
        history = []

        with tqdm(total=self.epochs) as p_bar:
            for _ in range(self.epochs):
                self.model.train()
                total_loss = 0

                for x, _ in train:
                    self.optimizer.zero_grad()

                    _, decoded = self.model(x)

                    loss = self.loss_fun(decoded, x)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / len(train)

                train_metrics = AutoEncoderMetrics(train_loss)

                progress_postfix = {
                    "loss": train_loss,
                }

                eval_metrics = None

                if eval is not None:
                    eval_metrics = self.eval(eval)

                    progress_postfix["eval_loss"] = eval_metrics.loss

                history.append(
                    AutoEncoderTrainerHistoryEntry(train_metrics, eval_metrics)
                )

                p_bar.set_postfix(**progress_postfix)
                p_bar.update()

        return TrainFeedBack(history)

    def eval(self, loader: DataLoader) -> AutoEncoderMetrics:
        self.model.eval()

        total_loss = 0

        for x, _ in loader:
            self.optimizer.zero_grad()
            _, decoded = self.model(x)
            loss = self.loss_fun(decoded, x)

            total_loss += loss.item()

        return AutoEncoderMetrics(loss=total_loss / len(loader))
