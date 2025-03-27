from dataclasses import dataclass, field
import numpy as np
import torch
import torch.utils.data as data

from torch import nn
from torch import optim
from collections import defaultdict
from sklearn.cluster import KMeans
from torchmetrics import Accuracy, CohenKappa, F1Score
from src.util.list_ext import group_indices
from src.util.patches import extract_label_patches, slice_and_patch
from src.util.progress_bar import create_progress_bar


@dataclass
class SpatialRegulatedSelfTrainerArgs:
    model: nn.Module
    optimizer: optim.Optimizer
    num_classes: int
    over_cluster_count: int
    over_cluster_count_decay: float
    loss_fun: nn.Module = nn.CrossEntropyLoss()
    num_epochs: int = 12
    splits: int = 4
    patch_size: int = 9
    init_patch_size: int = 5
    semantic_threshold: float = 0.5
    spatial_threshold: int = 8
    spatial_constraint_weights: list[float] = field(default_factory=lambda _: [1, 0.5])


class SpatialRegulatedSelfTrainer:

    def __init__(self, args: SpatialRegulatedSelfTrainerArgs, device):
        self.args = args
        self.cluster_counts = np.power(
            args.over_cluster_count,
            np.arange(args.num_epochs + 1, 0, -1) / args.over_cluster_count_decay,
        ).astype(int)
        self.pred_attempts = []
        self.device = device
        self.history = []
        self.f1 = F1Score(
            task="multiclass", num_classes=args.num_classes, average="macro"
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.kappa = CohenKappa(task="multiclass", num_classes=args.num_classes)

    def train(self, image, initial_labels, eval_labels=None):
        original_shape = initial_labels.shape
        init_patches, labels = self.init_slice_and_patch(image, initial_labels)

        with create_progress_bar()(total=4) as pb:
            pb.set_description("[INIT] Extract initial features")
            x = self.extract_init_features(init_patches)
            pb.update()
            pb.set_description(
                f"[INIT] Clustering features over {self.cluster_counts[0]} clusters"
            )
            k = self.over_cluster(self.cluster_counts[0], x)
            pb.update()
            pb.set_description("[INIT] Introducing semantic constraint")
            c = self.introduce_semantic_constraint_and_merge(k, labels)
            pb.update()
            pb.set_description("[INIT] Introducing spatial constraint")
            y = self.introduce_spatial_constraint(c, original_shape)
            pb.update()

        self.pred_attempts.append(y)

        z = self.slice_and_patch(image)

        with create_progress_bar()(range(len(self.cluster_counts) - 1)) as pb:
            for cluster_size in self.cluster_counts[1:]:
                x, loss = self.extract_features(z, y)
                k = self.over_cluster(cluster_size, x)
                c = self.introduce_semantic_constraint_and_merge(k, labels)
                y = self.introduce_spatial_constraint(c, original_shape)

                self.pred_attempts.append(y)

                progress = {
                    "loss": loss,
                }

                if eval_labels is not None:
                    metrics = self.eval_y(y, eval_labels)
                    progress = progress | metrics

                self.history.append(progress)

                pb.set_postfix(**progress)
                pb.update()

        return y

    def extract_features(self, z, y):
        y_true = torch.tensor(
            extract_label_patches(y), device=self.device, dtype=torch.float32
        )
        x = []
        train_total_loss = 0

        for z_i in z:
            x_i = self.args.model(z_i)
            x.append(x_i.detach().cpu().numpy())

            loss = self.args.loss_fun(x_i, y_true)

            train_total_loss += loss.item()

            self.args.optimizer.zero_grad()
            loss.backward()

            self.args.optimizer.step()

        loss = train_total_loss / len(z)

        return x, loss

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
            torch.tensor(patches[i], device=self.device, dtype=torch.float32).permute(
                0, 3, 1, 2
            )
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

    def over_cluster(self, cluster_size, features):
        return [
            KMeans(n_clusters=cluster_size).fit_predict(features[i])
            for i in range(len(features))
        ]

    def introduce_semantic_constraint_and_merge(self, clusters, labels):
        c = [self.introduce_semantic_constraint(it, labels) for it in clusters]
        return self.merge_clustering_results(c)

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

        def get_from_y(i, j):
            if i < 0 or i >= height or j < 0 or j >= width:
                return 0
            else:
                return y_matrix[i][j]

        result = np.zeros(original_shape)

        for i in range(height):
            for j in range(width):
                if y_matrix[i][j] > 0:
                    result[i][j] = y_matrix[i][j]
                else:
                    labels_count = defaultdict(int)

                    for radius, weight in enumerate(
                        self.args.spatial_constraint_weights
                    ):
                        for k in range(j - radius, j + radius + 1):
                            up = get_from_y(i, k - radius)
                            if up > 0:
                                labels_count[up] = labels_count[up] + weight
                            down = get_from_y(i, k + radius)
                            if down > 0:
                                labels_count[down] = labels_count[down] + weight

                        for k in range(i - radius + 1, i + radius):
                            left = get_from_y(i - radius, k)
                            if left > 0:
                                labels_count[left] = labels_count[left] + weight
                            right = get_from_y(i + radius, k)
                            if down > 0:
                                labels_count[right] = labels_count[right] + weight

                    pseudo_label = max(labels_count, key=labels_count.get, default=None)

                    if labels_count[pseudo_label] > self.args.spatial_threshold:
                        result[i][j] = pseudo_label

        return result
