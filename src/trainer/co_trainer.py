from typing import Optional

from torch import Generator, nn
import torch
from torch.utils import data

from src.data.dataset_decorator import UnlabeledDatasetDecorator
from src.model.ensemble import Ensemble
from src.trainer.base_trainer import BaseTrainer, TrainerFeedback, TrainerHistoryEntry
from src.util.progress_bar import create_progress_bar


class BiCoTrainer:

    def __init__(
        self,
        batch_size: int,
        confidence_threshold: float,
        generator: Generator,
        trainer: BaseTrainer,
    ):
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.generator = generator
        self.trainer = trainer

    def fit(
        self,
        models: tuple[nn.Module, nn.Module],
        labeled: data.Dataset,
        unlabeled: data.Dataset,
        eval_dl: Optional[data.DataLoader],
    ) -> TrainerFeedback:
        history = []
        model_1, model_2 = models
        ds_1, ds_2 = self.__split(labeled)

        run = True

        with create_progress_bar()() as pb:
            while run:
                self.trainer.fit(model_1, self.__to_dataloader(ds_1))
                self.trainer.fit(model_2, self.__to_dataloader(ds_2))

                if unlabeled:
                    x, new_ds_2, confidence_1 = self.__pseudo_label(model_1, unlabeled)
                    _, new_ds_1, confidence_2 = self.__pseudo_label(model_2, unlabeled)

                    ds_1 = data.ConcatDataset([ds_1, new_ds_1])
                    ds_2 = data.ConcatDataset([ds_2, new_ds_2])

                    used_mask = confidence_1 | confidence_2
                    remaining_mask = ~used_mask

                    unlabeled = data.TensorDataset(x[remaining_mask])
                else:
                    run = False

                t_metrics = {"unlabeled_len": len(unlabeled)}
                e_metrics = self.validate(models, eval_dl) if eval_dl else {}
                h_entry = TrainerHistoryEntry(t_metrics, e_metrics)

                history.append(h_entry)
                pb.set_postfix(**h_entry.as_postfix())
                pb.update()

        return TrainerFeedback(history)

    def validate(
        self, models: tuple[nn.Module, nn.Module], eval_dl: Optional[data.DataLoader]
    ) -> dict[str, float]:
        model_1, model_2 = models
        return self.trainer.validate(Ensemble([model_1, model_2]), eval_dl)

    def __pseudo_label(self, model, ds):
        x_ls, y_pred_ls = self.trainer.predict(
            model, self.__to_dataloader(UnlabeledDatasetDecorator(ds))
        )
        x = torch.cat(x_ls).cpu()
        y_pred = torch.cat(y_pred_ls).cpu()
        coef, idx = torch.max(y_pred, dim=1)
        high_confidence = coef > self.confidence_threshold

        return (
            x,
            data.TensorDataset(x[high_confidence], idx[high_confidence]),
            high_confidence,
        )

    def __to_dataloader(self, ds):
        return data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
        )

    def __split(self, ds):
        return data.random_split(ds, [0.5, 0.5], self.generator)
