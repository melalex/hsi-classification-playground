from abc import ABC
import math
from pathlib import Path
from torch import nn

from src.definitions import MODELS_FOLDER
from src.util.torch import load_model, save_model


class ModelStorage(ABC):

    def update(self, model: nn.Module, metrics: dict[str, float]):
        pass

    def get_best(self) -> nn.Module:
        pass


class NoopModelStorage(ModelStorage):

    def update(self, model: nn.Module, metrics: dict[str, float]):
        pass

    def get_best(self):
        return None


class SimpleFileBackedModelStorage(ModelStorage):

    def __init__(
        self, optimize_metric: str, model_name: str, models_folder: Path = MODELS_FOLDER
    ):
        super().__init__()

        self.optimize_metric = optimize_metric
        self.models_folder = models_folder
        self.model_name = model_name
        self.best_score = -math.inf

    def update(self, model: nn.Module, metrics: dict[str, float]):
        current_score = metrics[self.optimize_metric]

        if current_score > self.best_score:
            save_model(
                model=model, model_name=self.model_name, base_path=self.models_folder
            )

            self.best_score = current_score

    def get_best(self):
        return load_model(model_name=self.model_name, base_path=self.models_folder)
