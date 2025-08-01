from pathlib import Path
from sklearn.pipeline import make_pipeline
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from torchmetrics import Accuracy, CohenKappa, F1Score

from src.definitions import CACHE_FOLDER
from src.pipeline.common import (
    Clustering,
    DimensionalityReducer,
    FeatureExtractor,
    KMeansClustering,
    introduce_semantic_constraint,
    introduce_spatial_constraint,
)
from src.util.hsi import extract_image_patches, extract_label_patches, extract_patches, scale_image
from src.util.progress_bar import create_progress_bar


@dataclass
class ImprovedSpatialRegulatedSelfTrainingMetrics:
    overall_accuracy: Optional[float] = None
    average_accuracy: Optional[float] = None
    kappa_score: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class ImprovedSpatialRegulatedSelfTrainingStepSnapshot:
    extracted_features: list[np.array]
    clustering_result: list[np.array]
    semantic_constraint: list[np.array]
    spatial_constraint_result: np.array


@dataclass
class ImprovedSpatialRegulatedSelfTrainingHistoryEntry:
    feature_extractor_loss: float
    metrics: Optional[ImprovedSpatialRegulatedSelfTrainingMetrics]
    step_snapshots: Optional[ImprovedSpatialRegulatedSelfTrainingStepSnapshot]


@dataclass
class ImprovedSpatialRegulatedSelfTrainingPipelineArgs:
    num_classes: int
    cluster_sizes: list[int]
    feature_extractor: FeatureExtractor
    dim_reduction: DimensionalityReducer
    clustering: Clustering = KMeansClustering()
    patch_size: int = 9
    init_patch_size: int = 5
    semantic_threshold: float = 0.5
    spatial_threshold: int = 8
    spatial_constraint_weights: list[float] = field(default_factory=lambda: [1, 0.5])
    record_step_snapshots: bool = True
    cache_folder: Path = None
    verbose: bool = True


class ImprovedSpatialRegulatedSelfTrainingPipeline:
    num_classes: int
    cluster_sizes: list[int]
    feature_extractor: FeatureExtractor
    clustering: Clustering
    dim_reduction: DimensionalityReducer
    patch_size: int
    init_patch_size: int
    semantic_threshold: float
    spatial_threshold: int
    spatial_constraint_weights: list[float]
    history: list[ImprovedSpatialRegulatedSelfTrainingHistoryEntry]
    cache_folder: Path
    verbose: bool

    def __init__(self, args: ImprovedSpatialRegulatedSelfTrainingPipelineArgs, device):
        self.num_classes = args.num_classes
        self.cluster_sizes = args.cluster_sizes
        self.feature_extractor = args.feature_extractor
        self.dim_reduction = args.dim_reduction
        self.clustering = args.clustering
        self.patch_size = args.patch_size
        self.init_patch_size = args.init_patch_size
        self.semantic_threshold = args.semantic_threshold
        self.spatial_threshold = args.spatial_threshold
        self.spatial_constraint_weights = args.spatial_constraint_weights
        self.record_step_snapshots = args.record_step_snapshots
        self.cache_folder = args.cache_folder
        self.device = device
        self.verbose = args.verbose
        self.history = []

        self.f1 = F1Score(
            task="multiclass", num_classes=args.num_classes, average="weighted"
        )
        self.overall_accuracy = Accuracy(
            task="multiclass", num_classes=args.num_classes, average="macro"
        )
        self.average_accuracy = Accuracy(
            task="multiclass", num_classes=args.num_classes, average="micro"
        )
        self.kappa = CohenKappa(task="multiclass", num_classes=args.num_classes)

    def fit(self, image, initial_labels, eval_y=None):
        scaler, image = scale_image(image)

        self.scaler = scaler

        image = self.dim_reduction.fit(image)
        init_y = self.init_fit(image, initial_labels, eval_y)
        y = self.iter_fit(image, init_y, eval_y)

        return y

    def init_fit(self, image, initial_labels, eval_labels=None):
        original_shape = initial_labels.shape
        z, y = self.init_extract_patch(image, initial_labels)

        with create_progress_bar()(total=5, disable=not self.verbose) as pb:
            pb.set_description("[INIT] Extract initial features")
            x = self.extract_init_features(z)
            pb.update()
            pb.set_description(
                f"[INIT] Clustering features over {self.cluster_sizes[0]} clusters"
            )
            k = self.init_over_cluster(x)
            pb.update()
            pb.set_description("[INIT] Introducing semantic constraint")
            c = self.introduce_semantic_constraint(k, y)
            pb.update()
            pb.set_description("[INIT] Merge clustering results")
            pb.update()
            pb.set_description("[INIT] Introducing spatial constraint")
            y = self.introduce_spatial_constraint(c, original_shape)
            pb.update()

        metrics = self.eval_y(y, eval_labels)
        step_snapshot = ImprovedSpatialRegulatedSelfTrainingStepSnapshot(
            extracted_features=x,
            clustering_result=k,
            semantic_constraint=c,
            spatial_constraint_result=y,
        )

        self.record_history(None, metrics, step_snapshot)

        return y

    def iter_fit(self, image, init_y, eval_y):
        original_shape = init_y.shape
        z = self.extract_patches(image)
        y = init_y

        with create_progress_bar()(
            range(len(self.cluster_sizes) - 1), disable=not self.verbose
        ) as pb:
            for cluster_size in self.cluster_sizes[1:]:
                y = extract_label_patches(y)
                y_tensor = torch.tensor(y, device=self.device, dtype=torch.long)
                loss = self.feature_extractor.fit([z], y_tensor)
                x = self.feature_extractor.predict([z])[0]
                k = self.over_cluster(cluster_size, x)
                c = self.introduce_semantic_constraint(k, y)
                y = self.introduce_spatial_constraint(c, original_shape)

                if eval_y is not None:
                    metrics = self.eval_y(y, eval_y)
                    progress = {
                        "val_f1": metrics.f1_score,
                        "val_average_accuracy": metrics.average_accuracy,
                        "val_overall_accuracy": metrics.overall_accuracy,
                        "val_kappa": metrics.kappa_score,
                    }
                else:
                    metrics = None
                    progress = {}

                step_snapshot = ImprovedSpatialRegulatedSelfTrainingStepSnapshot(
                    extracted_features=[x],
                    clustering_result=[k],
                    semantic_constraint=[c],
                    spatial_constraint_result=y,
                )

                self.record_history(loss, metrics, step_snapshot)

                pb.set_postfix(**progress)
                pb.update()

        return y

    def extract_init_features(self, patches):
        return patches.reshape(patches.shape[0], -1)

    def init_extract_patch(self, image, initial_labels):
        return extract_patches(image, initial_labels, self.init_patch_size)

    def init_over_cluster(self, features):
        if self.cache_folder is None:
            return self.over_cluster(self.cluster_sizes[0], features)

        cache_loc = (
            self.cache_folder
            / f"{self.cluster_sizes[0]}-{self.dim_reduction.get_n_components()}-{len(self.cluster_sizes)}"
        )

        if cache_loc.exists():
            return np.load(cache_loc / "0.npy")

        k = self.over_cluster(self.cluster_sizes[0], features)

        cache_loc.mkdir(exist_ok=True, parents=True)

        np.save(cache_loc / "0", k)

        return k

    def record_history(self, loss, metrics, step_snapshot):
        if self.record_step_snapshots:
            self.history.append(
                ImprovedSpatialRegulatedSelfTrainingHistoryEntry(
                    loss, metrics, step_snapshot
                )
            )
        else:
            self.history.append(
                ImprovedSpatialRegulatedSelfTrainingHistoryEntry(loss, metrics, None)
            )

    def eval_y(self, y_pred, y_true):
        if y_true is None:
            return None

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        y_pred_arr = y_pred.ravel()
        y_true_arr = y_true.ravel()

        return ImprovedSpatialRegulatedSelfTrainingMetrics(
            overall_accuracy=self.overall_accuracy(y_pred_arr, y_true_arr).item(),
            average_accuracy=self.average_accuracy(y_pred_arr, y_true_arr).item(),
            kappa_score=self.kappa(y_pred_arr, y_true_arr).item(),
            f1_score=self.f1(y_pred_arr, y_true_arr).item(),
        )

    def extract_patches(self, image):
        patches = extract_image_patches(image=image, patch_size=self.patch_size)
        return torch.tensor(
            patches,
            device=self.device,
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

    def over_cluster(self, cluster_size, features):
        return self.clustering.cluster(cluster_size, features)

    def introduce_semantic_constraint(self, cluster, labels):
        return introduce_semantic_constraint(
            cluster, labels, self.num_classes, self.semantic_threshold
        )

    def introduce_spatial_constraint(self, c, original_shape):
        return introduce_spatial_constraint(
            c, original_shape, self.spatial_constraint_weights, self.spatial_threshold
        )
