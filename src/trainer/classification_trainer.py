import torch

from typing import Optional
from torch import Tensor, nn

from torch.nn import functional
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CohenKappa, F1Score

from src.data.dataset_decorator import UnlabeledDatasetDecorator
from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.util.progress_bar import create_progress_bar


class ClassificationTrainer(BaseTrainer):

    def __init__(
        self,
        num_epochs: int,
        num_classes: int,
        criterion: nn.Module,
        device: torch.device,
        record_history: bool = True,
        validate_every_n_steps: int = 1,
    ):
        self.num_epochs = num_epochs
        self.record_history = record_history
        self.criterion = criterion
        self.device = device
        self.validate_every_n_steps = validate_every_n_steps

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
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> TrainerFeedback:
        optimizer = model.configure_optimizers()
        scheduler = model.configure_scheduller(optimizer)

        history = []

        with create_progress_bar()(total=self.num_epochs) as pb:
            for epoch in range(self.num_epochs):
                if scheduler:
                    scheduler.step()

                model.train()

                train_total_loss = 0

                for x, y_true in train_dataloader:
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    optimizer.zero_grad()

                    y_pred = model(x)

                    loss = self.criterion(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                    train_total_loss += loss.item()

                epoch_loss = train_total_loss / len(train_dataloader)

                train_metrics = {"train_loss": epoch_loss}

                eval_metrics = (
                    self.validate(model, eval_dataloader)
                    if eval_dataloader and (epoch + 1) % self.validate_every_n_steps == 0
                    else {}
                )

                h_entry = TrainerHistoryEntry(train_metrics, eval_metrics)

                if self.record_history:
                    history.append(h_entry)

                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history)

    def predict(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        model.eval()

        result_x = []
        result_y = []

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)

                y_pred = model(x)

                result_x.append(x)
                result_y.append(y_pred)

        return result_x, result_y

    def validate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y_true in dataloader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = model(x)
                all_preds.append(y_pred)
                all_targets.append(y_true)

            y_pred_tensor = torch.cat(all_preds, dim=0)
            y_true_tensor = torch.cat(all_targets, dim=0)
            y_pred_classes = torch.argmax(y_pred_tensor, dim=1)

            f1 = self.f1(y_pred_classes, y_true_tensor).item()
            acc_overall = self.overall_accuracy(y_pred_classes, y_true_tensor).item()
            acc_avg = self.average_accuracy(y_pred_classes, y_true_tensor).item()
            kappa_score = self.kappa(y_pred_classes, y_true_tensor).item()
            loss = self.criterion(y_pred_tensor, y_true_tensor).item()

            return {
                "eval_f1": f1,
                "eval_accuracy_overall": acc_overall,
                "eval_accuracy_avg": acc_avg,
                "eval_kappa": kappa_score,
                "eval_loss": loss,
            }
