from abc import ABC
from typing import Optional
from torch import Tensor, nn
import torch
from torchmetrics import Accuracy, CohenKappa, F1Score
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.trainer.masked_autoencoder_trainer import Maksking
from src.trainer.model_storage import ModelStorage, NoopModelStorage


class MaskedAutoEncoderWithClasificationTrainer(BaseTrainer):

    def __init__(
        self,
        loss_fun: nn.Module,
        epochs: int,
        num_classes: int,
        masking: Maksking,
        device,
        extract_prediction=lambda y_pred: torch.argmax(y_pred, dim=1),
        validate_every_n_steps=1,
        model_storage: ModelStorage = NoopModelStorage(),
    ):
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.device = device
        self.validate_every_n_steps = validate_every_n_steps
        self.masking = masking
        self.extract_prediction = extract_prediction
        self.model_storage = model_storage

        self.f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        ).to(device)
        self.overall_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        self.average_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        ).to(device)
        self.kappa = CohenKappa(task="multiclass", num_classes=num_classes).to(device)

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
                total_cls_loss = 0
                batch_count = len(train)

                for x, y_true in train:
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    x_masked, mask = self.masking.mask(x)

                    optimizer.zero_grad()

                    _, decoded, y_pred = model(x_masked)

                    _, cls_loss, loss = self.loss_fun(decoded, x, mask, y_pred, y_true)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_cls_loss += cls_loss.item()

                train_loss = total_loss / batch_count
                train_cls_loss = total_cls_loss / batch_count

                train_metrics = {
                    "train_loss": train_loss,
                    "train_cls_loss": train_cls_loss,
                }

                eval_metrics = (
                    self.validate(model, eval)
                    if eval and (epoch + 1) % self.validate_every_n_steps == 0
                    else {}
                )

                self.model_storage.update(model, eval_metrics)

                h_entry = TrainerHistoryEntry(train_metrics, eval_metrics)

                history.append(h_entry)

                if scheduler:
                    scheduler.step()

                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history), self.model_storage.get_best()

    def validate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()

        total_loss = 0

        with torch.no_grad():
            for x, y_true in loader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                _, _, y_pred = model(x)
                _, loss, _ = self.loss_fun(
                    x, x, torch.ones(x.shape, device=self.device), y_pred, y_true
                )

                y_pred_classes = self.extract_prediction(y_pred)

                self.f1.update(y_pred_classes, y_true)
                self.overall_accuracy.update(y_pred_classes, y_true)
                self.average_accuracy.update(y_pred_classes, y_true)
                self.kappa.update(y_pred_classes, y_true)

                total_loss += loss.item()

            f1 = self.f1.compute().item()
            acc_overall = self.overall_accuracy.compute().item()
            acc_avg = self.average_accuracy.compute().item()
            kappa_score = self.kappa.compute().item()
            loss = total_loss / len(loader)

            self.f1.reset()
            self.overall_accuracy.reset()
            self.average_accuracy.reset()
            self.kappa.reset()

            return {
                "eval_f1": f1,
                "eval_accuracy_overall": acc_overall,
                "eval_accuracy_avg": acc_avg,
                "eval_kappa": kappa_score,
                "eval_loss": loss,
            }

    def predict(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        model.eval()

        result_x = []
        result_y = []

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)

                _, _, y_pred = model(x)

                result_x.append(x)
                result_y.append(y_pred)

        return result_x, result_y
