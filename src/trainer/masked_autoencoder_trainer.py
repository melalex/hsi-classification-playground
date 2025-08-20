from abc import ABC
from typing import Optional
from torch import Tensor, nn
import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.util.masking import Maksking


class MaskedAutoEncoderTrainer(BaseTrainer):

    def __init__(
        self,
        loss_fun: nn.Module,
        epochs: int,
        masking: Maksking,
        device,
        validate_every_n_steps=1,
    ):
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.device = device
        self.validate_every_n_steps = validate_every_n_steps
        self.masking = masking

    def fit(
        self,
        model: TrainableModule,
        train: DataLoader,
        eval: Optional[DataLoader] = None,
    ) -> TrainerFeedback:
        history = []
        model = model.to(self.device)
        optimizer = model.configure_optimizers()
        scheduler = model.configure_scheduler(optimizer)

        with tqdm(total=self.epochs) as pb:
            for epoch in range(self.epochs):
                model.train()
                total_loss = 0

                for x, _ in train:
                    x = x.to(self.device)
                    x_masked, mask = self.masking.mask(x)

                    optimizer.zero_grad()

                    _, decoded = model(x_masked)

                    loss = self.loss_fun(decoded, x, mask)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / len(train)

                train_metrics = {"train_loss": train_loss}

                eval_metrics = (
                    self.validate(model, eval)
                    if eval and (epoch + 1) % self.validate_every_n_steps == 0
                    else {}
                )

                h_entry = TrainerHistoryEntry(train_metrics, eval_metrics)

                history.append(h_entry)

                if scheduler:
                    scheduler.step()

                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history)

    def validate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()

        total_loss = 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)

                _, decoded = model(x)
                loss = self.loss_fun(
                    decoded, x, torch.ones(x.shape, device=self.device)
                )

                total_loss += loss.item()

        return {"eval_loss": total_loss / len(loader)}

    def predict(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        model.eval()

        result_x = []
        result_y = []

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)

                _, decoded = model(x)

                result_x.append(x)
                result_y.append(decoded)

        return result_x, result_y
