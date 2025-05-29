from typing import Optional

from torch import Generator, Tensor, nn
import torch
from torch.utils import data

from src.data.dataset_decorator import UnlabeledDatasetDecorator
from src.model.ensemble import MultiViewEnsemble
from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.trainer.model_support import ModelSupport
from src.util.progress_bar import create_progress_bar


class MultiViewTrainer:

    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        confidence_threshold: float,
        generator: Generator,
        device: torch.device,
        max_epochs: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.generator = generator
        self.support = ModelSupport(num_classes, device)
        self.max_epochs = max_epochs

    def fit(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: list[data.Dataset],
        unlabeled: data.Dataset,
        model_eval_dl: Optional[list[data.DataLoader]] = None,
        ensemble_eval_dl: Optional[data.DataLoader] = None,
    ) -> tuple[TrainerFeedback, nn.Module]:
        assert len(models) == len(trainers) and len(models) == len(labeled)

        history = []
        localy_labeled = labeled[:]
        models_count = len(models)
        model_eval_dl = model_eval_dl if model_eval_dl else [None] * len(models)

        run = True

        with create_progress_bar()() as pb:
            while run:
                run = False
                for model, trainer, ds, eval_dl in zip(
                    models, trainers, localy_labeled, model_eval_dl
                ):
                    trainer.fit(model, self.__to_dataloader(ds), eval_dl)

                if unlabeled:
                    for i, (model, trainer) in enumerate(zip(models, trainers)):
                        x, new_label, labeled_mask = self.__pseudo_label(
                            model, trainer, unlabeled
                        )

                        if len(new_label) > 0:
                            run = True

                            for j in range(models_count):
                                if j != i:
                                    localy_labeled[j] = data.ConcatDataset(
                                        [localy_labeled[j], new_label]
                                    )

                            unlabeled = data.TensorDataset(x[~labeled_mask])

                t_metrics = {"unlabeled_len": len(unlabeled)}
                e_metrics = (
                    self.validate(MultiViewEnsemble(models), ensemble_eval_dl)
                    if ensemble_eval_dl
                    else {}
                )
                h_entry = TrainerHistoryEntry(t_metrics, e_metrics)

                history.append(h_entry)
                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history), MultiViewEnsemble(models)

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

    def __pseudo_label(self, model: nn.Module, trainer: BaseTrainer, ds: data.Dataset):
        x_ls, y_pred_ls = trainer.predict(
            model, self.__to_dataloader(UnlabeledDatasetDecorator(ds))
        )
        x = torch.cat(x_ls).cpu()
        y_pred = torch.cat(y_pred_ls).cpu()
        high_confidence = y_pred > self.confidence_threshold
        labeled = x[high_confidence]

        return (
            x,
            data.TensorDataset(labeled, torch.zeros(labeled.shape[0])),
            high_confidence,
        )

    def __to_dataloader(self, ds, shuffle=True):
        return data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=self.generator,
        )
