from sklearn.discriminant_analysis import StandardScaler
import torch
import lightning
import numpy as np

from torch import nn
from abc import ABC
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from sklearn.cluster import KMeans
from torchmetrics import Accuracy, CohenKappa, F1Score
from torch.utils import data

from src.definitions import CACHE_FOLDER
from src.trainer.classification_trainer import ClassificationTrainer
from src.util.list_ext import group_indices
from src.util.patches import (
    extract_image_patches,
    extract_label_patches,
    scale_patched,
    slice_and_patch,
)
from src.util.progress_bar import create_progress_bar


class FeatureExtractor(ABC):

    def fit(self, z: list[torch.Tensor], y: torch.Tensor) -> float:
        pass

    def predict(self, z: list[torch.Tensor]) -> list[torch.Tensor]:
        pass


class CnnFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        model: nn.Module,
        trainer: ClassificationTrainer,
        generator: torch.Generator,
        batch_size: int = 64,
    ):
        self.model = model
        self.trainer = trainer
        self.batch_size = batch_size
        self.generator = generator

    def fit(self, z: list[torch.Tensor], y: torch.Tensor) -> float:
        train_loader = data.DataLoader(
            dataset=data.TensorDataset(torch.cat(z, dim=0), y.repeat(len(z))),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
        )

        return self.trainer.fit(self.model, train_loader)

    def predict(self, z: list[torch.Tensor]) -> list[torch.Tensor]:
        self.model.eval()
        return [self.model(z_i).detach().cpu().numpy() for z_i in z]


class MultipleCnnFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        models: list[nn.Module],
        trainer: ClassificationTrainer,
        generator: torch.Generator,
        batch_size: int = 64,
    ):
        self.models = models
        self.trainer = trainer
        self.batch_size = batch_size
        self.generator = generator

    def fit(self, z: list[torch.Tensor], y: torch.Tensor) -> float:
        acc_loss = 0

        for i, z_i in enumerate(z):
            train_loader = data.DataLoader(
                dataset=data.TensorDataset(z_i, y),
                batch_size=self.batch_size,
                shuffle=True,
                generator=self.generator,
            )

            acc_loss += self.trainer.fit(self.models[i], train_loader)

        return acc_loss / len(z)

    def predict(self, z: list[torch.Tensor]) -> list[torch.Tensor]:
        def get_model(i):
            model = self.models[i]
            model.eval()
            return model

        return [get_model(i)(z_i).detach().cpu().numpy() for i, z_i in enumerate(z)]


class FlatteningFeatureExtractor(FeatureExtractor):

    def fit(self, z: list[torch.Tensor], y: torch.Tensor) -> float:
        return 0

    def predict(self, z: list[torch.Tensor]) -> list[torch.Tensor]:
        return [z_i.reshape(z_i.shape[0], -1).detach().cpu().numpy() for z_i in z]


class Clustering(ABC):

    def cluster(self, num_clusters, x):
        pass


class KMeansClustering(Clustering):

    def __init__(self, seed=42):
        self.seed = seed

    def cluster(self, num_clusters, x):
        return KMeans(
            n_clusters=num_clusters, n_init=25, random_state=self.seed
        ).fit_predict(x)


@dataclass
class SpatialRegulatedSelfTrainingMetrics:
    overall_accuracy: Optional[float] = None
    average_accuracy: Optional[float] = None
    kappa_score: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class StepSnapshot:
    extracted_features: list[np.array]
    clustering_result: list[np.array]
    semantic_constraint: list[np.array]
    merged_semantic_constraint: np.array
    spatial_constraint_result: np.array


@dataclass
class HistoryEntry:
    feature_extractor_loss: float
    metrics: Optional[SpatialRegulatedSelfTrainingMetrics]
    step_snapshots: Optional[StepSnapshot]


@dataclass
class SpatialRegulatedSelfTrainingPipelineArgs:
    num_classes: int
    cluster_sizes: list[int]
    feature_extractor: FeatureExtractor
    clustering: Clustering = KMeansClustering()
    splits: int = 4
    patch_size: int = 9
    init_patch_size: int = 5
    semantic_threshold: float = 0.5
    spatial_threshold: int = 8
    spatial_constraint_weights: list[float] = field(default_factory=lambda: [1, 0.5])
    record_step_snapshots: bool = True


class SpatialRegulatedSelfTrainingPipeline:
    num_classes: int
    cluster_sizes: list[int]
    feature_extractor: FeatureExtractor
    clustering: Clustering
    splits: int
    patch_size: int
    init_patch_size: int
    semantic_threshold: float
    spatial_threshold: int
    spatial_constraint_weights: list[float]
    history: list[HistoryEntry]

    def __init__(self, args: SpatialRegulatedSelfTrainingPipelineArgs, device):
        self.num_classes = args.num_classes
        self.cluster_sizes = args.cluster_sizes
        self.feature_extractor = args.feature_extractor
        self.clustering = args.clustering
        self.splits = args.splits
        self.patch_size = args.patch_size
        self.init_patch_size = args.init_patch_size
        self.semantic_threshold = args.semantic_threshold
        self.spatial_threshold = args.spatial_threshold
        self.spatial_constraint_weights = args.spatial_constraint_weights
        self.record_step_snapshots = args.record_step_snapshots
        self.device = device
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
        scaler, image = self.scale_image(image)
        init_y = self.init_fit(image, initial_labels, eval_y)
        y = self.iter_fit(image, init_y, eval_y)

        return scaler, y

    def init_fit(self, image, initial_labels, eval_labels=None):
        original_shape = initial_labels.shape
        z, y = self.init_slice_and_patch(image, initial_labels)

        with create_progress_bar()(total=5) as pb:
            pb.set_description("[INIT] Extract initial features")
            x = self.extract_init_features(z)
            pb.update()
            pb.set_description(
                f"[INIT] Clustering features over {self.cluster_sizes[0]} clusters"
            )
            k = self.init_over_cluster(x)
            pb.update()
            pb.set_description("[INIT] Introducing semantic constraint")
            c = self.all_introduce_semantic_constraint(k, y)
            pb.update()
            pb.set_description("[INIT] Merge clustering results")
            c_m = self.merge_clustering_results(c)
            pb.update()
            pb.set_description("[INIT] Introducing spatial constraint")
            y = self.introduce_spatial_constraint(c_m, original_shape)
            pb.update()

        metrics = self.eval_y(y, eval_labels)
        step_snapshot = StepSnapshot(
            extracted_features=x,
            clustering_result=k,
            semantic_constraint=c,
            merged_semantic_constraint=c_m,
            spatial_constraint_result=y,
        )

        self.record_history(None, metrics, step_snapshot)

        return y

    def iter_fit(self, image, init_y, eval_y):
        original_shape = init_y.shape
        z = self.slice_and_patch(image)
        y = init_y

        with create_progress_bar()(range(len(self.cluster_sizes) - 1)) as pb:
            for cluster_size in self.cluster_sizes[1:]:
                y = extract_label_patches(y)
                y_tensor = torch.tensor(y, device=self.device, dtype=torch.long)
                loss = self.feature_extractor.fit(z, y_tensor)
                x = self.feature_extractor.predict(z)
                k = self.over_cluster(cluster_size, x)
                c = self.all_introduce_semantic_constraint(k, y)
                c_m = self.merge_clustering_results(c)
                y = self.introduce_spatial_constraint(c_m, original_shape)

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

                step_snapshot = StepSnapshot(
                    extracted_features=x,
                    clustering_result=k,
                    semantic_constraint=c,
                    merged_semantic_constraint=c_m,
                    spatial_constraint_result=y,
                )

                self.record_history(loss, metrics, step_snapshot)

                pb.set_postfix(**progress)
                pb.update()

        return y

    def scale_image(self, image):
        h, w, c = image.shape
        img_reshaped = image.reshape(-1, c)
        scaler = StandardScaler()
        img_scaled = scaler.fit_transform(img_reshaped)

        return scaler, img_scaled.reshape(h, w, c)

    def extract_init_features(self, patches):
        return [
            patches[i].reshape(patches[i].shape[0], -1) for i in range(len(patches))
        ]

    def init_slice_and_patch(self, image, initial_labels):
        init_patches = slice_and_patch(
            image=image,
            patch_size=self.init_patch_size,
            splits=self.splits,
        )
        labels = extract_label_patches(initial_labels)
        return init_patches, labels

    def init_over_cluster(self, features):
        cache_loc = (
            CACHE_FOLDER
            / "init-clustering"
            / f"{self.cluster_sizes[0]}-{self.splits}-{len(self.cluster_sizes)}"
        )

        if cache_loc.exists():
            return [np.load(cache_loc / f"{i}.npy") for i in range(self.splits)]

        k = self.over_cluster(self.cluster_sizes[0], features)

        cache_loc.mkdir(exist_ok=True, parents=True)

        for i, k_i in enumerate(k):
            np.save(cache_loc / f"{i}", k_i)

        return k

    def record_history(self, loss, metrics, step_snapshot):
        if self.record_step_snapshots:
            self.history.append(HistoryEntry(loss, metrics, step_snapshot))
        else:
            self.history.append(HistoryEntry(loss, metrics, None))

    def eval_y(self, y_pred, y_true):
        if y_true is None:
            return None

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        y_pred_arr = y_pred.ravel()
        y_true_arr = y_true.ravel()

        return SpatialRegulatedSelfTrainingMetrics(
            overall_accuracy=self.overall_accuracy(y_pred_arr, y_true_arr).item(),
            average_accuracy=self.average_accuracy(y_pred_arr, y_true_arr).item(),
            kappa_score=self.kappa(y_pred_arr, y_true_arr).item(),
            f1_score=self.f1(y_pred_arr, y_true_arr).item(),
        )

    def slice_and_patch(self, image):
        patches = slice_and_patch(
            image=image,
            patch_size=self.patch_size,
            splits=self.splits,
        )

        return [
            torch.tensor(
                patches[i],
                device=self.device,
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
            for i in range(len(patches))
        ]

    def over_cluster(self, cluster_size, features):
        return [
            self.clustering.cluster(cluster_size, features[i])
            for i in range(len(features))
        ]

    def all_introduce_semantic_constraint(self, clusters, labels):
        return [self.introduce_semantic_constraint(it, labels) for it in clusters]

    def introduce_semantic_constraint(self, cluster, labels):
        cluster_to_index = group_indices(cluster)
        cluster_sum = {}

        for cluster_id, elements in cluster_to_index.items():
            cluster_sum[cluster_id] = sum([1 for it in elements if labels[it] > 0])

        cluster_avg = sum(cluster_sum.values()) / len(cluster_to_index)

        purity_max = {}

        for f in range(1, self.num_classes):
            for cluster_id, elements in cluster_to_index.items():
                s_i = cluster_sum[cluster_id]
                if s_i > cluster_avg:
                    s_pt = sum([1 for it in elements if labels[it] == f])
                    pure_f = s_pt / s_i

                    if pure_f > self.semantic_threshold:
                        purity_max[cluster_id] = f

        result = np.zeros(len(labels))

        for cluster_id in cluster_to_index:
            if cluster_id in purity_max:
                f = purity_max[cluster_id]
                for elem in cluster_to_index[cluster_id]:
                    result[elem] = f

        return result

    def merge_clustering_results(self, results):
        stacked = np.vstack(results)

        same_values = np.all(stacked == stacked[0, :], axis=0)

        return np.where(same_values, stacked[0, :], 0)

    def introduce_spatial_constraint(self, c, original_shape):
        height, width = original_shape
        y_matrix = c.reshape(original_shape)

        result = np.zeros(original_shape)

        for i in range(height):
            for j in range(width):
                if y_matrix[i][j] > 0:
                    result[i][j] = y_matrix[i][j]
                else:
                    result[i][j] = self.calculate_pseudo_label(i, j, y_matrix)

        return result

    def calculate_pseudo_label(self, i, j, y):
        height, width = y.shape

        def get_from_y(i, j):
            if i < 0 or i >= height or j < 0 or j >= width:
                return 0
            else:
                return y[i][j]

        labels_count = defaultdict(int)

        for r, weight in enumerate(self.spatial_constraint_weights):
            radius = r + 1
            for k in range(j - radius, j + radius + 1):
                up = get_from_y(i - radius, k)
                if up > 0:
                    labels_count[up] += weight
                down = get_from_y(i + radius, k)
                if down > 0:
                    labels_count[down] += weight

            for k in range(i - radius + 1, i + radius):
                left = get_from_y(k, j - radius)
                if left > 0:
                    labels_count[left] += weight
                right = get_from_y(k, j + radius)
                if right > 0:
                    labels_count[right] += weight

        pseudo_label = max(labels_count, key=labels_count.get, default=None)

        if labels_count[pseudo_label] > self.spatial_threshold:
            return pseudo_label
        else:
            return 0
