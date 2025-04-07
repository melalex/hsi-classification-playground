from torch import optim, nn
import lightning as L
import torch
from torchmetrics import CohenKappa
from torchmetrics.classification import F1Score
from torchmetrics.classification import Accuracy

from src.model.fully_convolutional_lenet import FullyConvolutionalLeNet


class HyperSpectralImageClassifier(L.LightningModule):
    def __init__(self, net: nn.Module, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.net = net
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
