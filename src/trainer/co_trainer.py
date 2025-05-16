from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch import Generator, nn
import torch
from torch.utils import data
from torchmetrics import Accuracy, CohenKappa, F1Score

from src.data.dataset_decorator import UnlabeledDatasetDecorator
from src.model.ensemble import Ensemble
from src.trainer.classification_trainer import ClassificationTrainer
from src.util.progress_bar import create_progress_bar


@dataclass
class BiCoTrainerHistoryEntry:
    train: dict[str, float] = field(default_factory=lambda: {})
    eval: dict[str, float] = field(default_factory=lambda: {})


@dataclass
class BiCoTrainerFeedBack:
    history: list[BiCoTrainerHistoryEntry]


class BiCoTrainer:

    def __init__(
        self,
        batch_size: int,
        confidence_threshold: float,
        generator: Generator,
        cpu_count: int,
        trainer,
    ):
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.generator = generator
        self.cpu_count = cpu_count
        self.trainer = trainer

    def fit(
        self,
        models: tuple[nn.Module, nn.Module],
        model_wrappers: tuple[
            Callable[[nn.Module], object], Callable[[nn.Module], object]
        ],
        labeled: data.Dataset,
        unlabeled: data.Dataset,
        eval_dl: Optional[data.DataLoader],
    ):
        history = []
        init_max_epochs = self.trainer.fit_loop.max_epochs

        model_1, model_2 = models
        wrapper_1, wrapper_2 = model_wrappers
        ds_1, ds_2 = self.__split(labeled)

        run = True

        with create_progress_bar()() as pb:
            while run:
                wrapped_model_1 = wrapper_1(model_1)
                wrapped_model_2 = wrapper_2(model_2)

                self.trainer.fit(wrapped_model_1, self.__to_dataloader(ds_1))
                self.trainer.fit_loop.max_epochs += init_max_epochs

                self.trainer.fit(wrapped_model_2, self.__to_dataloader(ds_2))
                self.trainer.fit_loop.max_epochs += init_max_epochs

                if unlabeled:
                    new_ds_2, new_ds_1, unlabeled = self.__pseudo_label(
                        wrapped_model_1, wrapped_model_2, unlabeled
                    )

                    ds_1 = data.ConcatDataset([ds_1, new_ds_1])
                    ds_2 = data.ConcatDataset([ds_2, new_ds_2])
                else:
                    run = False

                pb_postfix = {"unlabeled_len": len(unlabeled)}

                if eval_dl:
                    eval_target = wrapper_1(Ensemble([model_1, model_2]))
                    eval_metrics = self.trainer.validate(eval_target, eval_dl)

                    pb_postfix.update(eval_metrics[0])

                    history.append(BiCoTrainerHistoryEntry(eval=eval_metrics))

                pb.set_postfix(**pb_postfix)
                pb.update()

        return BiCoTrainerFeedBack(history)

    def __pseudo_label(self, model_1, model_2, ds):
        def pseudo_label_step(model, x):
            y_pred = model(x)
            coef, idx = torch.max(y_pred, dim=1)
            high_confidence = coef > self.confidence_threshold

            return idx, high_confidence

        pseudo_labeled_x_1 = []
        pseudo_labeled_y_1 = []

        pseudo_labeled_x_2 = []
        pseudo_labeled_y_2 = []

        unlabeled = []

        with torch.no_grad():
            for x in self.__to_dataloader(UnlabeledDatasetDecorator(ds)):
                y_pred_1, high_confidence_1 = pseudo_label_step(model_1, x)
                y_pred_2, high_confidence_2 = pseudo_label_step(model_2, x)

                pseudo_labeled_x_1.append(x[high_confidence_1])
                pseudo_labeled_y_1.append(y_pred_1[high_confidence_1])
                pseudo_labeled_x_2.append(x[high_confidence_2])
                pseudo_labeled_y_2.append(y_pred_2[high_confidence_2])

                unlabeled.append(x[~high_confidence_1 & ~high_confidence_2])

        return (
            data.TensorDataset(
                torch.cat(pseudo_labeled_x_1), torch.cat(pseudo_labeled_y_1)
            ),
            data.TensorDataset(
                torch.cat(pseudo_labeled_x_2), torch.cat(pseudo_labeled_y_2)
            ),
            data.TensorDataset(torch.cat(unlabeled)),
        )

    def __to_dataloader(self, ds):
        return data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cpu_count,
            persistent_workers=True,
            generator=self.generator,
        )

    def __split(self, ds):
        return data.random_split(ds, [0.5, 0.5], self.generator)
