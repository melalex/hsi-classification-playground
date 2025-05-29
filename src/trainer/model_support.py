import torch

from typing import Optional
from torch import Tensor, nn
from torch.utils import data
from torchmetrics import Accuracy, CohenKappa, F1Score

from src.trainer.classification_trainer import ClassificationTrainer


class ModelSupport:

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
    ):
        self.device = device

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

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        model.to(self.device)
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

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        model.to(self.device)
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

            f1 = self.f1(y_pred_tensor, y_true_tensor).item()
            acc_overall = self.overall_accuracy(y_pred_tensor, y_true_tensor).item()
            acc_avg = self.average_accuracy(y_pred_tensor, y_true_tensor).item()
            kappa_score = self.kappa(y_pred_tensor, y_true_tensor).item()

            return {
                "eval_f1": f1,
                "eval_accuracy_overall": acc_overall,
                "eval_accuracy_avg": acc_avg,
                "eval_kappa": kappa_score,
            }
