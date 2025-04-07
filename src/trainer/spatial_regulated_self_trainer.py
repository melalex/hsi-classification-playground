from dataclasses import dataclass, field
import pickle
from typing import Optional
import numpy as np
import torch
import torch.utils.data as data

from torch import nn
from torch import optim
from collections import defaultdict
from sklearn.cluster import KMeans
from torchmetrics import Accuracy, CohenKappa, F1Score
from src.model.fully_convolutional_lenet import FullyConvolutionalLeNet
from src.util.list_ext import group_indices
from src.util.patches import extract_label_patches, scale_patched, slice_and_patch
from src.util.progress_bar import create_progress_bar


@dataclass
class SpatialRegulatedSelfTrainerArgs:
    model: nn.Module
    optimizer: optim.Optimizer
    num_classes: int
    over_cluster_count: int
    over_cluster_count_decay: float
    loss_fun: nn.Module = nn.CrossEntropyLoss()
    cnn_train_epochs: int = 12
    cnn_train_batch_size: int = 34
    record_by_step_results: bool = True
    num_epochs: int = 12
    splits: int = 4
    patch_size: int = 9
    init_patch_size: int = 5
    semantic_threshold: float = 0.5
    spatial_threshold: int = 8
    spatial_constraint_weights: list[float] = field(default_factory=lambda: [1, 0.5])


@dataclass
class SpatialRegulatedSelfTrainingMetrics:
    loss: Optional[float] = None
    overall_accuracy: Optional[float] = None
    kappa_score: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class StepResults:
    extracted_features: list[np.array]
    clustering_result: list[np.array]
    semantic_constraint: list[np.array]
    merged_semantic_constraint: np.array
    spatial_constraint_result: np.array


@dataclass
class HistoryEntry:
    metrics: Optional[SpatialRegulatedSelfTrainingMetrics]
    by_step_results: Optional[StepResults]


class SpatialRegulatedSelfTrainer:
    history: list[HistoryEntry]

    def __init__(self, args: SpatialRegulatedSelfTrainerArgs, lr, device):
        self.args = args
        # self.cluster_sizes = args.over_cluster_count * np.exp(
        #     np.arange(args.num_epochs, 0, -1) * args.over_cluster_count_decay,
        # ).astype(int)
        self.cluster_sizes = np.array(
            [
                18248,
                16730.17,
                15212.33,
                13694.50,
                12176.67,
                10658.83,
                9141.00,
                7623.17,
                6105.33,
                4587.50,
                3069.67,
                1551.83,
                34.00,
            ]
        ).astype(int)
        self.init_cluster_size = self.cluster_sizes[0]
        self.cluster_sizes = self.cluster_sizes[1:]
        self.device = device
        self.history = []
        self.f1 = F1Score(
            task="multiclass", num_classes=args.num_classes, average="macro"
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.kappa = CohenKappa(task="multiclass", num_classes=args.num_classes)
        self.models_arr = [
            FullyConvolutionalLeNet(50, 17).to(device),
            FullyConvolutionalLeNet(50, 17).to(device),
            FullyConvolutionalLeNet(50, 17).to(device),
            FullyConvolutionalLeNet(50, 17).to(device),
        ]
        self.optimizers_arr = [
            optim.Adam(self.models_arr[0].parameters(), lr=lr),
            optim.Adam(self.models_arr[1].parameters(), lr=lr),
            optim.Adam(self.models_arr[2].parameters(), lr=lr),
            optim.Adam(self.models_arr[3].parameters(), lr=lr),
        ]

    def train(self, image, initial_labels, eval_labels=None):
        original_shape = initial_labels.shape
        init_patches, labels = self.init_slice_and_patch(image, initial_labels)
        metrics = None
        by_step_results = None

        with create_progress_bar()(total=5) as pb:
            pb.set_description("[INIT] Extract initial features")
            x = self.extract_init_features(init_patches)
            pb.update()
            pb.set_description(
                f"[INIT] Clustering features over {self.init_cluster_size} clusters"
            )
            k = self.init_over_cluster(x)
            pb.update()
            pb.set_description("[INIT] Introducing semantic constraint")
            c = self.all_introduce_semantic_constraint(k, labels)
            pb.update()
            pb.set_description("[INIT] Merge clustering results")
            c_m = self.merge_clustering_results(c)
            pb.update()
            pb.set_description("[INIT] Introducing spatial constraint")
            y = self.introduce_spatial_constraint(c_m, original_shape)
            pb.update()

        if eval_labels is not None:
            metrics_dict = self.eval_y(y, eval_labels)
            metrics = SpatialRegulatedSelfTrainingMetrics(
                overall_accuracy=metrics_dict["val_f1"],
                kappa_score=metrics_dict["val_accuracy"],
                f1_score=metrics_dict["val_kappa"],
            )

        if self.args.record_by_step_results:
            by_step_results = StepResults(
                extracted_features=x,
                clustering_result=k,
                semantic_constraint=c,
                merged_semantic_constraint=c_m,
                spatial_constraint_result=y,
            )

        self.append_history(metrics, by_step_results)

        z = init_patches

        with create_progress_bar()(range(len(self.cluster_sizes))) as pb:
            for cluster_size in self.cluster_sizes:
                loss = self.train_feature_extractor(z, y)
                x = self.extract_features(z)
                k = self.over_cluster(cluster_size, x)
                c = self.all_introduce_semantic_constraint(k, labels)
                c_m = self.merge_clustering_results(c)
                y = self.introduce_spatial_constraint(c_m, original_shape)

                progress = {
                    "loss": loss,
                }

                if eval_labels is not None:
                    metrics_dict = self.eval_y(y, eval_labels)
                    progress = progress | metrics_dict
                    metrics = SpatialRegulatedSelfTrainingMetrics(
                        loss=loss,
                        overall_accuracy=metrics_dict["val_f1"],
                        kappa_score=metrics_dict["val_accuracy"],
                        f1_score=metrics_dict["val_kappa"],
                    )
                else:
                    metrics = SpatialRegulatedSelfTrainingMetrics(loss=loss)

                if self.args.record_by_step_results:
                    by_step_results = StepResults(
                        extracted_features=x,
                        clustering_result=k,
                        semantic_constraint=c,
                        merged_semantic_constraint=c_m,
                        spatial_constraint_result=y,
                    )

                self.append_history(metrics, by_step_results)

                pb.set_postfix(**progress)
                pb.update()

        return y

    def append_history(
        self,
        metrics: Optional[SpatialRegulatedSelfTrainingMetrics],
        by_step_results: Optional[StepResults],
    ):
        self.history.append(HistoryEntry(metrics, by_step_results))

    def train_feature_extractor(self, z, y):
        return 0
        y_tensor = torch.tensor(
            extract_label_patches(y), device=self.device, dtype=torch.float32
        )
        train_total_loss = 0

        for _ in range(self.args.cnn_train_epochs):
            for i, z_i in enumerate(z):
                self.models_arr[i].train()
                train_loader = data.DataLoader(
                    data.TensorDataset(z_i, y_tensor),
                    batch_size=self.args.cnn_train_batch_size,
                    shuffle=True,
                )

                for x, y_true in train_loader:
                    self.optimizers_arr[i].zero_grad()

                    y_pred = self.models_arr[i](x)

                    loss = self.args.loss_fun(y_pred, y_true)
                    loss.backward()
                    self.optimizers_arr[i].step()

                    train_total_loss += loss.item()

        return train_total_loss / (len(z) * self.args.cnn_train_epochs)

    def extract_features(self, z):
        return self.extract_init_features(z)
        result = []
        for i, z_i in enumerate(z):
            self.models_arr[i].eval()
            x_i = self.models_arr[i](z_i)
            result.append(x_i.detach().cpu().numpy())

        return result

    def init_slice_and_patch(self, image, initial_labels):
        init_patches = slice_and_patch(
            image=image,
            patch_size=self.args.init_patch_size,
            splits=self.args.splits,
        )
        labels = extract_label_patches(initial_labels)
        return init_patches, labels

    def slice_and_patch(self, image):
        patches = slice_and_patch(
            image=image,
            patch_size=self.args.patch_size,
            splits=self.args.splits,
        )

        return [
            torch.tensor(
                scale_patched(patches[i])[1],
                #  device=self.device,
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
            for i in range(patches.shape[0])
        ]

    def eval_y(self, y_pred, y_true):
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        y_pred_arr = y_pred.ravel()
        y_true_arr = y_true.ravel()
        return {
            "val_f1": self.f1(y_pred_arr, y_true_arr).item(),
            "val_accuracy": self.accuracy(y_pred_arr, y_true_arr).item(),
            "val_kappa": self.kappa(y_pred, y_true).item(),
        }

    def extract_init_features(self, patches):
        return [
            patches[i, :, :, :].reshape(patches.shape[1], -1)
            for i in range(patches.shape[0])
        ]

    def init_over_cluster(self, features):
        return [
            np.load(
                f"/Users/alexandermelashchenko/Workspace/spatial-regulated-self-training/tmp/{i}.npy"
            )
            for i in range(self.args.splits)
        ]
        k = self.over_cluster(self.init_cluster_size, features)

        for i, k_i in enumerate(k):
            np.save(
                f"/Users/alexandermelashchenko/Workspace/spatial-regulated-self-training/tmp/single-{i}",
                k_i,
            )

        return k

    def over_cluster(self, cluster_size, features):
        return [
            KMeans(n_clusters=cluster_size).fit_predict(features[i])
            for i in range(len(features))
        ]

    def introduce_semantic_constraint_and_merge(self, clusters, labels):
        c = [self.introduce_semantic_constraint(it, labels) for it in clusters]
        return self.merge_clustering_results(c)

    def all_introduce_semantic_constraint(self, clusters, labels):
        return [self.introduce_semantic_constraint(it, labels) for it in clusters]

    def introduce_semantic_constraint(self, cluster, labels):
        cluster_to_index = group_indices(cluster)
        cluster_sum = {}

        for cluster_id, elements in cluster_to_index.items():
            cluster_sum[cluster_id] = sum([1 for it in elements if labels[it] > 0])

        cluster_avg = sum(cluster_sum.values()) / len(cluster_to_index)

        purity_max_f = defaultdict(int)
        purity_max = defaultdict(int)

        for f in range(1, self.args.num_classes):
            for cluster_id, elements in cluster_to_index.items():
                s_i = cluster_sum[cluster_id]
                if s_i > cluster_avg:
                    s_pt = sum([1 for it in elements if labels[it] == f])
                    pure_f = s_pt / s_i

                    if (
                        pure_f > self.args.semantic_threshold
                        and pure_f > purity_max[cluster_id]
                    ):
                        purity_max_f[cluster_id] = f
                        purity_max[cluster_id] = pure_f

        result = np.zeros(len(labels))

        for cluster_id in cluster_to_index:
            if cluster_id in purity_max_f:
                f = purity_max_f[cluster_id]
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

        for r, weight in enumerate(self.args.spatial_constraint_weights):
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
                if down > 0:
                    labels_count[right] += weight

        pseudo_label = max(labels_count, key=labels_count.get, default=None)

        if labels_count[pseudo_label] >= self.args.spatial_threshold:
            return pseudo_label
        else:
            return 0
