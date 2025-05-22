import torch

from torch import nn

from src.definitions import CACHE_FOLDER
from src.model.lenet import FullyConvolutionalLeNet
from src.model.greed_search import GreedSearchAdapter
from src.pipeline.common import CnnFeatureExtractor, KMeansClustering
from src.pipeline.spatial_regulated_self_training_pipeline import SpatialRegulatedSelfTrainingPipeline, SpatialRegulatedSelfTrainingPipelineArgs
from src.trainer.base_trainer import AdamOptimizedModule
from src.trainer.classification_trainer import ClassificationTrainer
from src.util.list_ext import divide_interval


class SpatialRegulatedSelfTrainingPipelineGreedSearchAdapter(
    GreedSearchAdapter[SpatialRegulatedSelfTrainingPipeline]
):

    def __init__(
        self,
        image,
        masked_labels,
        labels,
        num_classes,
        device,
        cpu_count,
        random_seed,
        generator,
    ):
        self.image = image
        self.masked_labels = masked_labels
        self.labels = labels
        self.num_classes = num_classes
        self.device = device
        self.random_seed = random_seed
        self.generator = generator
        self.cpu_count = cpu_count

    def init_interval(self) -> list[tuple[int, int]]:
        return divide_interval(1, 210, self.cpu_count)

    def get_params(self, i: int) -> dict[str, float]:
        return {"k_values": [i * 100]}

    def init_model(self, split, params: dict[str, float]):
        splits = 4
        patch_size = 9
        init_patch_size = 5
        semantic_threshold = 0.6
        spatial_threshold = 8
        spatial_constraint_weights = [1, 0.5]
        feature_extractor_num_epochs = 100
        feature_extractor_learning_rate = 1e-5

        torch.cuda.empty_cache()

        _, _, c = self.image.shape

        input_channels = int(c / splits)

        model = AdamOptimizedModule(
            FullyConvolutionalLeNet(input_channels, self.num_classes),
            lr=feature_extractor_learning_rate,
        )

        trainer = ClassificationTrainer(
            num_epochs=feature_extractor_num_epochs,
            num_classes=self.num_classes,
            device=self.device,
            criterion=nn.CrossEntropyLoss(),
        )

        args = SpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=self.num_classes,
            cluster_sizes=params["k_values"],
            feature_extractor=CnnFeatureExtractor(
                model, trainer, self.generator, batch_size=64
            ),
            clustering=KMeansClustering(seed=self.random_seed),
            splits=splits,
            patch_size=patch_size,
            init_patch_size=init_patch_size,
            semantic_threshold=semantic_threshold,
            spatial_threshold=spatial_threshold,
            spatial_constraint_weights=spatial_constraint_weights,
            cache_folder=CACHE_FOLDER / f"init-clustering-split-{split}",
            record_step_snapshots=True,
            verbose=False,
        )

        return SpatialRegulatedSelfTrainingPipeline(args, self.device)

    def fit_model(self, model: SpatialRegulatedSelfTrainingPipeline):
        model.fit(self.image, self.masked_labels, self.labels)

    def score_model(
        self, model: SpatialRegulatedSelfTrainingPipeline
    ) -> list[dict[str, float]]:
        return [it.metrics.__dict__ for it in model.history]
