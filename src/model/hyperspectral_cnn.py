from torch import optim, nn
import lightning as L
import torch
from torchmetrics.classification import F1Score


class HyperSpectralCnn(L.LightningModule):
    def __init__(self, input_channels, num_classes, lr=1e-3, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 100, kernel_size=4),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(100, 200, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(200),
            nn.Conv2d(200, 500, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(500, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], self.num_classes)
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

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)

        return {"val_loss": loss, "val_f1": f1}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
