from dataclasses import dataclass
from typing import Callable, Optional
from torch import optim, nn
import lightning as L
import torch
from torchmetrics import CohenKappa
from torchmetrics.classification import F1Score
from torchmetrics.classification import Accuracy


@dataclass
class HyperSpectralImageClassifierMetrics:
    loss: Optional[float] = None
    f1: Optional[float] = None
    overall_accuracy: Optional[float] = None
    average_accuracy: Optional[float] = None
    kappa: Optional[float] = None


class HyperSpectralImageClassifier(L.LightningModule):

    def __init__(
        self,
        net: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay=0,
        scheduler: Optional[Callable] = None,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.net = net
        self.scheduler = scheduler
        self.train_metrics = []
        self.val_metrics = []

        self.f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.overall_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.average_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.kappa = CohenKappa(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        prediction = torch.argmax(y_hat, dim=1)
        f1 = self.f1(prediction, y)
        overall_accuracy = self.overall_accuracy(prediction, y)
        average_accuracy = self.average_accuracy(prediction, y)
        kappa = self.kappa(prediction, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "val_overall_accuracy",
            overall_accuracy,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_average_accuracy",
            average_accuracy,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("val_kappa", kappa, prog_bar=True, on_epoch=True, on_step=False)

        return {
            "val_loss": loss,
            "val_f1": f1,
            "val_overall_accuracy": overall_accuracy,
            "val_average_accuracy": average_accuracy,
            "val_kappa": kappa,
        }

    def on_train_epoch_end(self):
        self.train_metrics.append(
            HyperSpectralImageClassifierMetrics(
                loss=self.trainer.callback_metrics.get("loss")
            )
        )

    def on_validation_epoch_end(self):
        self.val_metrics.append(
            HyperSpectralImageClassifierMetrics(
                loss=self.trainer.callback_metrics.get("val_loss"),
                f1=self.trainer.callback_metrics.get("val_f1"),
                overall_accuracy=self.trainer.callback_metrics.get(
                    "val_overall_accuracy"
                ),
                average_accuracy=self.trainer.callback_metrics.get(
                    "val_average_accuracy"
                ),
                kappa=self.trainer.callback_metrics.get("val_kappa"),
            )
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": None,
                },
            }
        else:
            return optimizer
