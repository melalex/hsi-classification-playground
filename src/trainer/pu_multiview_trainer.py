from collections import defaultdict
from typing import Optional

from torch import Generator, Tensor, nn
import torch
from torch.utils import data

from src.data.dataset_decorator import (
    NegativeDataset,
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
from src.util.torch import dataloader_from_prtototype


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
            pb.set_postfix(unlabeled_len=len(localy_unlabeled.dataset))

            for i in range(self.max_epochs):
                pb.set_description(f"Epoch {i}")

                early_stop = True

                self.__fit_all(
                    models=models,
                    trainers=trainers,
                    labeled=localy_labeled,
                )

                localy_unlabeled, localy_labeled, early_stop = self.__pseudo_label_all(
                    models=models,
                    labeled=localy_labeled,
                    unlabeled=localy_unlabeled,
                )

                t_metrics = {"unlabeled_len": len(localy_unlabeled.dataset)}
                e_metrics = self.validate(MultiViewEnsemble(models), eval_dl)
                h_entry = TrainerHistoryEntry(t_metrics, e_metrics)

                history.append(h_entry)
                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

                if early_stop:
                    break

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
                shuffle=False,
                generator=self.generator,
            )

            trainer.fit(model, dl)

    def __pseudo_label(self, model: nn.Module, x: Tensor):
        y_pred = self.support.predict_batch(model, x)
        return y_pred > self.confidence_threshold

    def __pseudo_label_all(
        self, models: list[TrainableModule], x: Tensor, y_i: list[Tensor]
    ):
        models_count = len(models)
        new_labeled = defaultdict(list)
        new_unlabled = []
        labels_count_not_increased = True

        high_confidence = [self.__pseudo_label(it, x) for it in models]
        stacked_confidence = torch.stack(high_confidence)
        only_one_confident = stacked_confidence.sum(dim=0) == 1
        unconfident = ~only_one_confident

        new_unlabled.append(x[unconfident.cpu()])

        for i, confident in enumerate(high_confidence):
            mask = confident & only_one_confident

            for j, y in enumerate(y_i):
                unlabeled = 
                if j != i:
                    y[mask] = 0
                else:
                    y[mask] = 1

        new_labeled_dl = []

        for i, it in enumerate(labeled):
            if new_labeled[i]:
                labels_count_not_increased = False
                new_labeled_dl.append(
                    dataloader_from_prtototype(
                        data.ConcatDataset([it.dataset, *new_labeled[i]]), it
                    )
                )
            else:
                new_labeled_dl.append(it)

        new_unlabeld_dl = dataloader_from_prtototype(
            UnlabeledDatasetDecorator(data.TensorDataset(torch.cat(new_unlabled))),
            unlabeled,
        )

        return new_unlabeld_dl, new_labeled_dl, labels_count_not_increased
