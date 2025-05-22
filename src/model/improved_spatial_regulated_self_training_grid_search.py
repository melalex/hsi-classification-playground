from typing import Sequence
import torch

from torch import nn

from src.definitions import CACHE_FOLDER
from src.model.lenet import FullyConvolutionalLeNet
from src.model.grid_search import GridSearchAdapter
from src.pipeline.common import (
    CnnFeatureExtractor,
    KMeansClustering,
    PcaDimensionalityReducer,
)
from src.pipeline.improved_regulated_self_training_pipeline import (
    ImprovedSpatialRegulatedSelfTrainingPipeline,
    ImprovedSpatialRegulatedSelfTrainingPipelineArgs,
)
from src.trainer.classification_trainer import ClassificationTrainer
from src.util.over_clustering import exponential_decay_over_clustering


class ImprovedSpatialRegulatedSelfTrainingPipelineGridSearchAdapter(
    GridSearchAdapter[ImprovedSpatialRegulatedSelfTrainingPipeline]
):

    def __init__(
        self,
        params,
        image,
        masked_labels,
        labels,
        num_classes,
        device,
        random_seed,
        generator,
    ):
        self.params = params
        self.image = image
        self.masked_labels = masked_labels
        self.labels = labels
        self.num_classes = num_classes
        self.device = device
        self.random_seed = random_seed
        self.generator = generator

    def params_grid(self) -> dict[str, Sequence[float]]:
        return self.params

    def init_model(self, split, params: dict[str, float]):
        torch.cuda.empty_cache()

        _, _, c = self.image.shape

        input_channels = params["input_channels"]

        model = FullyConvolutionalLeNet(input_channels, self.num_classes).to(
            self.device
        )

        trainer = ClassificationTrainer(
            num_epochs=params["feature_extractor_epochs"],
            learning_rate=params["learning_rate"],
            loss_fun=nn.CrossEntropyLoss(),
        )

        k_values = exponential_decay_over_clustering(
            k_star=params["k_star"],
            lambda_v=params["lambda_v"],
            max_iter=params["num_epochs"],
        )

        args = ImprovedSpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=self.num_classes,
            cluster_sizes=k_values,
            feature_extractor=CnnFeatureExtractor(
                model, trainer, self.generator, batch_size=params["batch_size"]
            ),
            dim_reduction=PcaDimensionalityReducer(input_channels),
            clustering=KMeansClustering(seed=self.random_seed),
            patch_size=params["patch_size"],
            init_patch_size=5,
            semantic_threshold=params["semantic_threshold"],
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
            record_step_snapshots=True,
            verbose=False,
            cache_folder=CACHE_FOLDER / f"improved-init-clustering-split-{split}",
        )

        return ImprovedSpatialRegulatedSelfTrainingPipeline(args, self.device)

    def fit_model(self, model: ImprovedSpatialRegulatedSelfTrainingPipeline):
        model.fit(self.image, self.masked_labels, self.labels)

    def score_model(
        self, model: ImprovedSpatialRegulatedSelfTrainingPipeline
    ) -> list[dict[str, float]]:
        return [it.metrics.__dict__ for it in model.history]
