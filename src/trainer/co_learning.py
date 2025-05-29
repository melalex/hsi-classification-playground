import torch

from typing import Optional
from torch import Tensor, nn
from torch.utils import data
from statsmodels.stats.proportion import proportion_confint
from torchmetrics import Accuracy, CohenKappa, F1Score

from src.data.dataset_decorator import LabeledDatasetDecorator
from src.model.democratic_model import DemocraticModel
from src.trainer.base_trainer import BaseTrainer, TrainableModule
from src.trainer.classification_trainer import ClassificationTrainer
from src.trainer.model_support import ModelSupport
from src.util.progress_bar import create_progress_bar
from src.util.torch import dataloader_from_prtototype


class DemocraticCoLearningTrainer:

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        confidence_interval_alpha: float = 0.05,
        max_epochs: Optional[int] = None,
    ):
        self.support = ModelSupport(num_classes, device)
        self.confidence_interval_alpha = confidence_interval_alpha
        self.max_epochs = max_epochs

    def fit(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: data.DataLoader,
        unlabeled: data.DataLoader,
        eval_dataloader: Optional[data.DataLoader],
    ) -> DemocraticModel:
        num_models = len(models)

        history = []

        originally_labeled = [labeled] * num_models
        labeled_dl = [labeled] * num_models
        estimated_errors = torch.tensor([0] * num_models)

        is_changed = True
        current_epoch = 0

        with create_progress_bar()(total=self.max_epochs) as pb:
            while is_changed and (
                self.max_epochs is None or current_epoch < self.max_epochs
            ):
                is_changed = False

                self.__train_learners(models, trainers, labeled_dl, eval_dataloader)
                predictions = self.__predict_unlabeled(models, trainers, unlabeled)
                majority_vote = self.__compute_majority_vote(predictions)

                original_confidence = self.__compute_confidence_intervals(
                    models, trainers, originally_labeled
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
                    desision_mask, labeled_dl, candidates, majority_vote, unlabeled
                )

                is_changed = torch.any(desision_mask)

                eval_model = self.__create_democratic_model(
                    models, trainers, originally_labeled
                )

                if eval_dataloader:
                    eval_result = self.validate(eval_model, eval_dataloader)
                    history.append(eval_result)
                    pb.set_postfix(**eval_result)

                pb.update()
                current_epoch += 1

        return history, self.__create_democratic_model(
            models, trainers, originally_labeled
        )

    def predict(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> tuple[list[Tensor], list[Tensor]]:
        return self.support.predict(model, dataloader)

    def validate(
        self, model: nn.Module, dataloader: data.DataLoader
    ) -> dict[str, float]:
        return self.support.validate(model, dataloader)

    def __train_learners(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: list[data.DataLoader],
        eval_dataloader: Optional[data.DataLoader],
    ):
        for model, trainer, dl in zip(models, trainers, labeled):
            trainer.fit(model, dl, test_dataloader=eval_dataloader)

    def __predict_unlabeled(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        unlabeld: data.DataLoader,
    ) -> list[list[Tensor]]:
        return [
            torch.argmax(torch.cat(trainer.predict(model, unlabeld)[1]), dim=1)
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
            _, y_true, y_pred = trainer.predict_labeled(model, dataloader)
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            y_pred = torch.argmax(y_pred, dim=1)

            trials = len(y_true)
            correct = torch.count_nonzero(y_true == y_pred)

            result.append(
                proportion_confint(
                    count=correct.cpu(),
                    nobs=trials,
                    alpha=self.confidence_interval_alpha,
                    method="wilson",
                )
            )

        return result

    def __choose_new_labels_candidates(
        self, majority_vote: Tensor, predictions: list[Tensor], weights: list[float]
    ) -> list[Tensor]:
        predictions_tensor = torch.stack(predictions)
        agreement_mask = predictions_tensor == majority_vote
        disagreement_mask = ~agreement_mask
        weights_tensor = torch.tensor(weights, device=agreement_mask.device).unsqueeze(
            1
        )

        agrees = agreement_mask.float() * weights_tensor
        disagrees = disagreement_mask.float() * weights_tensor

        agrees_weight = torch.sum(agrees, dim=0)

        disagrees_weight = torch.sum(disagrees, dim=0)

        candidates = agrees_weight > disagrees_weight

        disagreeing_candidates = disagreement_mask & candidates

        return [torch.where(row)[0] for row in disagreeing_candidates]

    def __estimate_accuracy_improvement(
        self,
        confidence: list[tuple[float, float]],
        estimated_errors: Tensor,
        estimator_labels: list[data.DataLoader],
        new_estimator_labels: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        estimator_labels_count = torch.tensor(
            [len(it.dataset) for it in estimator_labels]
        )
        new_estimator_labels_count = torch.tensor(
            [len(it) for it in new_estimator_labels]
        )

        num_models = len(estimator_labels_count)

        e_factor = 1 - sum([l for l, _ in confidence]) / num_models

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
                    majority_vote[indecies_to_add].cpu(),
                )

                dataloader = dataloader_from_prtototype(
                    data.ConcatDataset([old_ds, new_ds]), dataloader
                )

            result.append(dataloader)

        return result

    def __create_democratic_model(
        self,
        models: list[TrainableModule],
        trainers: list[BaseTrainer],
        labeled: list[data.DataLoader],
    ) -> DemocraticModel:
        confidence = self.__compute_confidence_intervals(models, trainers, labeled)
        weights = self.__compute_weights(confidence)

        result_models = [it for it, w in zip(models, weights) if w > 0.5]
        result_weight = [w for w in weights if w > 0.5]

        return DemocraticModel(result_models, result_weight)
