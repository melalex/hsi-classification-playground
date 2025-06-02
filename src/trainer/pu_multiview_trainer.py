from collections import defaultdict
from typing import Optional

from torch import Generator, Tensor, nn
import torch
from torch.utils import data

from src.data.dataset_decorator import (
    PuDatasetDecorator,
    UnlabeledDatasetDecorator,
)
from src.model.ensemble import MultiViewEnsemble
from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.trainer.model_support import ModelSupport
from src.util.progress_bar import create_progress_bar


class PuMultiViewTrainer:

    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        confidence_threshold: float,
        generator: Generator,
        device: torch.device,
        max_epochs: Optional[int] = None,
        ensemble_confidence_threshold: float = 0.5,
    ):
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.generator = generator
        self.support = ModelSupport(num_classes, device)
        self.max_epochs = max_epochs
        self.ensemble_confidence_threshold = ensemble_confidence_threshold

    def fit(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        x: Tensor,
        y: Tensor,
        eval_dl: Optional[data.DataLoader] = None,
    ) -> tuple[TrainerFeedback, nn.Module]:
        history = []
        y_i = [torch.clone(y) for _ in range(len(models))]

        with create_progress_bar()(total=self.max_epochs) as pb:
            pb.set_postfix(avg_unlabeled_len=self.__count_avg_unlebeled(y_i))

            for i in range(self.max_epochs):
                pb.set_description(f"Epoch {i}")

                self.__fit_all(models=models, trainers=trainers, x=x, y_i=y_i)

                y_i = self.__pseudo_label_all(models=models, x=x, y_i=y_i)

                t_metrics = {"avg_unlabeled_len": self.__count_avg_unlebeled(y_i)}
                e_metrics = self.validate(MultiViewEnsemble(models), eval_dl)
                h_entry = TrainerHistoryEntry(t_metrics, e_metrics)

                history.append(h_entry)
                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history), MultiViewEnsemble(
            models, self.ensemble_confidence_threshold
        )

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        return self.support.predict(model, dataloader)

    def validate(
        self,
        model: nn.Module,
        eval_dl: Optional[data.DataLoader],
    ) -> dict[str, float]:
        return self.support.validate(model, eval_dl)

    def __count_avg_unlebeled(self, y_i: list[Tensor]) -> float:
        return sum([torch.sum(it == -1) for it in y_i]) / len(y_i)

    def __fit_all(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        x: Tensor,
        y_i: list[Tensor],
    ):
        for i, (model, trainer, y) in enumerate(zip(models, trainers, y_i)):
            dl = data.DataLoader(
                dataset=PuDatasetDecorator(data.TensorDataset(x, y), i),
                batch_size=self.batch_size,
                shuffle=True,
                generator=self.generator,
            )

            trainer.fit(model, dl)

    def __pseudo_label(self, model: nn.Module, x: Tensor):
        dl = data.DataLoader(
            dataset=UnlabeledDatasetDecorator(data.TensorDataset(x)),
            batch_size=self.batch_size,
            shuffle=False,
            generator=self.generator,
        )

        _, y_pred = self.support.predict(model, dl)
        y_pred = torch.cat(y_pred).cpu()
        return y_pred > self.confidence_threshold

    def __pseudo_label_all(
        self,
        models: list[TrainableModule],
        x: Tensor,
        y_i: list[Tensor],
    ):
        high_confidence = [self.__pseudo_label(it, x) for it in models]
        stacked_confidence = torch.stack(high_confidence)
        only_one_confident = stacked_confidence.sum(dim=0) == 1

        for i, confident in enumerate(high_confidence):
            mask = confident & only_one_confident

            for j, y in enumerate(y_i):
                unlabeled = y == -1
                if j != i:
                    y[mask & unlabeled] = 0
                else:
                    y[mask & unlabeled] = 1

        return y_i
