import torch

from typing import Optional
from torch import Tensor, nn
from torch.utils import data
from torchmetrics import Accuracy, CohenKappa, F1Score


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

    def predict_batch(self, model: nn.Module, x: Tensor) -> Tensor:
        with torch.no_grad():
            model.to(self.device)
            model.eval()

            x = x.to(self.device)

            return model(x)

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        if not dataloader:
            return {}

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for x, y_true in dataloader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = model(x)

                self.f1.update(y_pred, y_true)
                self.overall_accuracy.update(y_pred, y_true)
                self.average_accuracy.update(y_pred, y_true)
                self.kappa.update(y_pred, y_true)

            f1 = self.f1.compute().item()
            acc_overall = self.overall_accuracy.compute().item()
            acc_avg = self.average_accuracy.compute().item()
            kappa_score = self.kappa.compute().item()

            self.f1.reset()
            self.overall_accuracy.reset()
            self.average_accuracy.reset()
            self.kappa.reset()

            return {
                "eval_f1": f1,
                "eval_accuracy_overall": acc_overall,
                "eval_accuracy_avg": acc_avg,
                "eval_kappa": kappa_score,
            }
