from abc import ABC
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils import data

from src.trainer.classification_trainer import ClassificationTrainer
from src.util.list_ext import group_indices


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
        if len(z) == 1:
            train_loader = data.DataLoader(
                dataset=data.TensorDataset(z[0], y),
                batch_size=self.batch_size,
                shuffle=True,
                generator=self.generator,
            )
        else:
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

    def __init__(self, seed=42, n_init=5):
        self.seed = seed
        self.n_init = n_init

    def cluster(self, num_clusters, x):
        return KMeans(
            n_clusters=num_clusters, n_init=self.n_init, random_state=self.seed
        ).fit_predict(x)


class DimensionalityReducer(ABC):

    def reduce(self, input: np.array) -> tuple[BaseEstimator, np.array]:
        pass

    def get_n_components(self) -> int:
        pass


class PcaDimensionalityReducer(DimensionalityReducer):

    def __init__(self, n_components):
        self.n_components = n_components

    def reduce(self, input: np.array) -> tuple[BaseEstimator, np.array]:
        h, w, c = input.shape
        reshaped_data = input.reshape(c, -1).T

        pca = PCA(n_components=self.n_components)
        reduced_data = pca.fit_transform(reshaped_data)

        reduced_image = reduced_data.T.reshape(h, w, self.n_components)

        return pca, reduced_image

    def get_n_components(self):
        return self.n_components


def introduce_semantic_constraint(cluster, labels, num_classes, semantic_threshold):
    cluster_to_index = group_indices(cluster)
    cluster_sum = {}

    for cluster_id, elements in cluster_to_index.items():
        cluster_sum[cluster_id] = sum([1 for it in elements if labels[it] > 0])

    cluster_avg = sum(cluster_sum.values()) / len(cluster_to_index)

    purity_max = {}

    for f in range(1, num_classes):
        for cluster_id, elements in cluster_to_index.items():
            s_i = cluster_sum[cluster_id]
            if s_i > cluster_avg:
                s_pt = sum([1 for it in elements if labels[it] == f])
                pure_f = s_pt / s_i

                if pure_f > semantic_threshold:
                    purity_max[cluster_id] = f

    result = np.zeros(len(labels))

    for cluster_id in cluster_to_index:
        if cluster_id in purity_max:
            f = purity_max[cluster_id]
            for elem in cluster_to_index[cluster_id]:
                result[elem] = f

    return result


def merge_clustering_results(results):
    stacked = np.vstack(results)

    same_values = np.all(stacked == stacked[0, :], axis=0)

    return np.where(same_values, stacked[0, :], 0)


def introduce_spatial_constraint(
    c, original_shape, spatial_constraint_weights, spatial_threshold
):
    height, width = original_shape
    y_matrix = c.reshape(original_shape)

    result = np.zeros(original_shape)

    for i in range(height):
        for j in range(width):
            if y_matrix[i][j] > 0:
                result[i][j] = y_matrix[i][j]
            else:
                result[i][j] = calculate_pseudo_label(
                    i, j, y_matrix, spatial_constraint_weights, spatial_threshold
                )

    return result


def calculate_pseudo_label(i, j, y, spatial_constraint_weights, spatial_threshold):
    height, width = y.shape

    def get_from_y(i, j):
        if i < 0 or i >= height or j < 0 or j >= width:
            return 0
        else:
            return y[i][j]

    labels_count = defaultdict(int)

    for r, weight in enumerate(spatial_constraint_weights):
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

    if labels_count[pseudo_label] > spatial_threshold:
        return pseudo_label
    else:
        return 0
