from torch import optim, nn
import lightning as L
import torch
from torchmetrics.classification import F1Score


class HyperSpectralCnn(L.LightningModule):
    def __init__(self, input_channels, num_classes, lr=1e-3, dropout=0.5):
        super().__init__()
        self.lr = lr
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 512, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.adapter = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adapter(x)
        x = self.classifier(x)
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
