from typing import Optional
from torch import Generator, Tensor
import torch
from torch.utils import data
import torch.utils
from statsmodels.stats.proportion import proportion_confint

from src.data.dataset_decorator import LabeledDatasetDecorator
from src.model.democratic_model import DemocraticModel
from src.trainer.base_trainer import BaseTrainer, TrainableModule, TrainerFeedback
from src.util.progress_bar import create_progress_bar


class DemocraticCoLearningTrainer:

    def __init__(
        self,
        confidence_interval_alpha: float,
    ):
        self.confidence_interval_alpha = confidence_interval_alpha

    def fit(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: data.DataLoader,
        unlabeld: data.DataLoader,
        eval_dataloader: Optional[data.DataLoader],
    ) -> TrainerFeedback:
        num_models = len(models)

        originaly_labeled = [labeled] * num_models
        labeled_dl = [labeled] * num_models
        estimated_errors = [0] * num_models

        is_changed = True

        with create_progress_bar()() as pb:
            while is_changed:
                is_changed = False

                self.__train_learners(models, trainers, labeled_dl)
                predictions = self.__predict_unlabled(models, trainers, unlabeld)
                majority_vote = self.__compute_majority_vote(predictions)

                original_confidence = self.__compute_confidence_intervals(
                    models, trainers, originaly_labeled
                )

                weights = self.__compute_weights(original_confidence)
                candidates = self.__choose_new_labels_candidates(
                    majority_vote, predictions, weights
                )

                confidence = self.__compute_confidence_intervals(
                    models, trainers, labeled_dl
                )

                desision_mask, estimated_errors = self.__estimate_accuracy_improvement(
                    confidence, estimated_errors, labeled_dl, candidates
                )

                labeled_dl = self.__update_labeled_dl(
                    desision_mask, labeled_dl, candidates, majority_vote, unlabeld
                )

                is_changed = torch.any(desision_mask)

                pb.update()

        return self.__create_democratic_model(models, trainers, labeled)

    def __train_learners(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: list[data.DataLoader],
    ):
        for model, trainer, dl in zip(models, trainers, labeled):
            trainer.fit(model, dl)

    def __predict_unlabled(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        unlabeld: data.DataLoader,
    ) -> list[list[Tensor]]:
        return [
            torch.argmax(torch.cat(trainer.predict(model, unlabeld)), dim=1)
            for model, trainer in zip(models, trainers)
        ]

    def __compute_majority_vote(self, predictions):
        prediction_table = torch.stack(predictions, dim=1)

        modes, _ = torch.mode(prediction_table, dim=1)

        return modes

    def __compute_weights(self, confidence: list[tuple[float, float]]) -> list[float]:
        return [(li + hi) / 2 for (li, hi) in confidence]

    def __compute_confidence_intervals(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        dataloaders: list[data.DataLoader],
    ) -> list[tuple[float, float]]:
        result = []

        for model, trainer, dataloader in zip(models, trainers, dataloaders):
            y_true, y_pred = trainer.predict_labeled(model, dataloader)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            y_pred = torch.argmax(y_pred, dim=1)

            trials = y_true[0]
            correct = torch.count_nonzero(y_true == y_pred)

            result.append(
                proportion_confint(
                    correct,
                    trials,
                    alpha=self.confidence_interval_alpha,
                    method="wilson",
                )
            )

        return result

    def __choose_new_labels_candidates(
        self, majority_vote: Tensor, predictions: list[Tensor], weights: list[float]
    ) -> list[Tensor]:
        predictions_tensor = torch.cat(predictions)
        agreement_mask = predictions_tensor == majority_vote
        disagreement_mask = ~agreement_mask
        weights_tensor = torch.tensor(weights, device=agreement_mask.device)

        agrees = agreement_mask.float() * weights_tensor
        disagrees = disagreement_mask.float() * weights_tensor

        agrees_weight = torch.sum(agrees, dim=0)
        disagrees_weight = torch.sum(disagrees, dim=0)
        candidates = agrees_weight > disagrees_weight

        return [torch.nonzero(row, as_tuple=True)[0] for row in candidates]

    def __estimate_accuracy_improvement(
        self,
        confidence: list[tuple[float, float]],
        estimated_errors: Tensor,
        estimator_labels: list[data.DataLoader],
        new_estimator_labels: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        num_models = len(estimator_labels_count)

        e_factor = 1 - sum([l for l, _ in confidence]) / num_models

        estimator_labels_count = torch.tensor(
            [len(it.dataset) for it in estimator_labels]
        )
        new_estimator_labels_count = torch.tensor(
            [len(it) for it in new_estimator_labels]
        )

        q = estimator_labels_count * (
            (1 - 2 * (estimated_errors / estimator_labels_count)) ** 2
        )
        new_estimated_errors = e_factor * new_estimator_labels_count
        new_q = (estimator_labels_count + new_estimator_labels_count) * (
            1
            - 2
            * (estimated_errors + new_estimated_errors)
            / (estimator_labels_count + new_estimator_labels_count)
        ) ** 2

        desision_mask = new_q > q
        estimated_errors = (
            estimated_errors + desision_mask.float() * new_estimated_errors
        )

        return desision_mask, estimated_errors

    def __update_labeled_dl(
        self,
        desision_mask,
        estimator_labels: list[data.DataLoader],
        new_estimator_labels: list[Tensor],
        majority_vote: Tensor,
        unlabeld: data.DataLoader,
    ) -> list[Tensor]:
        unlabeld_ds = unlabeld.dataset
        result = []

        for i in range(len(estimator_labels)):
            dataloader = estimator_labels[i]
            if desision_mask[i]:
                old_ds = dataloader.dataset
                indecies_to_add = new_estimator_labels[i]

                new_ds = LabeledDatasetDecorator(
                    data.Subset(unlabeld_ds, indecies_to_add),
                    majority_vote[indecies_to_add],
                )

                dataloader = self.__to_dataloader(
                    data.ConcatDataset([old_ds, new_ds]), dataloader
                )

            result.append(dataloader)

        return result

    def __create_democratic_model(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: data.DataLoader,
    ):
        confidence = self.__compute_confidence_intervals(models, trainers, labeled)
        weights = self.__compute_weights(confidence)

        result_models = [it for it, w in zip(models, weights) if w > 0.5]
        result_weight = [w for w in weights if w > 0.5]

        return DemocraticModel(result_models, result_weight)

    def __to_dataloader(
        self, ds: data.Dataset, dataloader: data.DataLoader
    ) -> data.DataLoader:
        return data.DataLoader(
            dataset=ds,
            batch_size=dataloader.batch_size,
            shuffle=True,
            sampler=dataloader.sampler,
            batch_sampler=dataloader.batch_sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=dataloader.generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device,
        )
