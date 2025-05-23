import numpy as np
import torch

from typing import Optional
from torch import Generator, Tensor, nn
from torch.utils import data

from src.data.dataset_decorator import UnlabeledDatasetDecorator
from src.pipeline.common import (
    Clustering,
    introduce_semantic_constraint,
    introduce_spatial_constraint,
)
from src.trainer.base_trainer import BaseTrainer, TrainableModule, TrainerFeedback
from src.util.progress_bar import create_progress_bar


class LabelPropagationTrainer:

    def __init__(
        self,
        trainer: BaseTrainer,
        k_values: list[int],
        clustering: Clustering,
        num_classes: int,
        generator: Generator,
        batch_size: int = 64,
        semantic_threshold: float = 0.5,
        spatial_constraint_weights: list[float] = [1, 0.5],
        spatial_threshold: float = 8,
    ):
        self.trainer = trainer
        self.k_values = k_values
        self.clustering = clustering
        self.num_classes = num_classes
        self.generator = generator
        self.batch_size = batch_size
        self.semantic_threshold = semantic_threshold
        self.spatial_constraint_weights = spatial_constraint_weights
        self.spatial_threshold = spatial_threshold

    def fit(
        self,
        model: TrainableModule,
        x: np.ndarray,
        y: np.ndarray,
        eval: Optional[data.DataLoader],
    ) -> list[TrainerFeedback]:
        history = []
        original_shape = y.shape
        y = y.flatten()

        with create_progress_bar()(total=len(self.k_values)) as pb:
            for k in self.k_values:
                b, _, _, _ = x.shape
                cluster = self.clustering.cluster(k, x.reshape(b, -1))

                y += 1
                y = introduce_semantic_constraint(
                    cluster, y, self.num_classes, self.semantic_threshold
                )
                y = introduce_spatial_constraint(
                    y,
                    original_shape,
                    self.spatial_constraint_weights,
                    self.spatial_threshold,
                )
                y -= 1

                labeled_mask = y > -1

                x_labeled = x[labeled_mask, :, :, :]
                y_labeled = y[labeled_mask]

                train_ds = data.TensorDataset(Tensor(x_labeled), Tensor(y_labeled))
                train_dl = data.DataLoader(
                    train_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    generator=self.generator,
                )

                feadback = self.trainer.fit(model, train_dl, eval)

                history.append(feadback)

                predict_dl = data.DataLoader(
                    UnlabeledDatasetDecorator(data.TensorDataset(Tensor(x))),
                    batch_size=self.batch_size
                )

                x, y = self.trainer.predict(model, predict_dl)

                x = torch.cat(x).cpu().numpy()
                y = torch.cat(y).cpu().numpy()

                pb.set_postfix(**feadback.history[-1].as_postfix())
                pb.update()

        return history
