from pathlib import Path
from torch import nn

from src.definitions import MODELS_FOLDER
from src.util.torch import load_model


class FineTuneModel(nn.Module):

    def __init__(
        self, base_model, base_model_out_dim, num_classes, flatten_out: bool = False
    ):
        super().__init__()

        self.base_model = base_model
        self.cls_head = nn.Linear(base_model_out_dim, num_classes)
        self.flatten_out = flatten_out

    def forward(self, x):
        x = self.base_model(x)
        x = self.cls_head(x)

        if self.flatten_out:
            return x.reshape(-1)

        return x


class SimpleBinClassificationHead(nn.Module):

    def __init__(self, from_dim: int):
        super().__init__()

        self.cls_head = nn.Linear(from_dim, 1)

    def forward(self, x):
        x = self.cls_head(x)

        return x.reshape(-1)


def crete_model_for_fine_tune(
    base_model_out_dim: int,
    num_classes: int,
    model_name: str,
    flatten_out: bool = False,
    base_path: Path = MODELS_FOLDER,
):
    return FineTuneModel(
        base_model=load_model(model_name, base_path),
        base_model_out_dim=base_model_out_dim,
        num_classes=num_classes,
        flatten_out=flatten_out,
    )
