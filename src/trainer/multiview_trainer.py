from typing import Optional

from torch import Generator, nn
import torch
from torch.utils import data

from src.data.dataset_decorator import BinaryDatasetDecorator, UnlabeledDatasetDecorator
from src.model.ensemble import MultiViewEnsemble
from src.trainer.base_trainer import (
    BaseTrainer,
    TrainableModule,
    TrainerFeedback,
    TrainerHistoryEntry,
)
from src.util.progress_bar import create_progress_bar


class MultiViewTrainer:

    def __init__(
        self,
        batch_size: int,
        confidence_threshold: float,
        generator: Generator,
    ):
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.generator = generator

    def fit(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: list[data.Dataset],
        unlabeled: data.Dataset,
        eval_ds_list: Optional[list[data.Dataset]] = None,
    ) -> TrainerFeedback:
        assert len(models) == len(trainers) and len(models) == len(labeled)

        history = []
        localy_labeled = labeled[:]
        models_count = len(models)
        eval_ds_list = eval_ds_list if eval_ds_list else [None]

        run = True

        with create_progress_bar()() as pb:
            while run:
                run = False
                for model, trainer, ds, eval_ds in zip(
                    models, trainers, localy_labeled, eval_ds_list
                ):
                    trainer.fit(
                        model,
                        self.__to_dataloader(ds),
                        self.__to_dataloader(eval_ds),
                    )

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
                e_metrics = {}
                h_entry = TrainerHistoryEntry(t_metrics, e_metrics)

                history.append(h_entry)
                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history)

    def validate(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        eval_dl: Optional[data.DataLoader],
    ) -> dict[str, float]:
        return trainers[1].validate(MultiViewEnsemble(models), eval_dl)

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
