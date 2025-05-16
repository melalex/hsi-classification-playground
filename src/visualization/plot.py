import numpy as np
import seaborn as sns

from matplotlib import animation, pyplot as plt

from src.pipeline.spatial_regulated_self_training_pipeline import (
    SpatialRegulatedSelfTrainingHistoryEntry,
)


def plot_epoch_generic(feedback: list[float], desc="Loss", size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(feedback, label=desc)
    plt.title(desc)
    plt.xlabel("Epoch")
    plt.ylabel(desc)
    plt.legend()
    plt.show()


def plot_segmentation_comparison(
    ground_truth,
    predicted_labels,
    title1="Ground Truth",
    title2="Predicted",
):
    num_classes = len(np.unique(ground_truth))
    # Define color map (seaborn deep to get distinguishable colors)
    cmap = plt.colormaps.get_cmap("jet")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Ground Truth
    axes[0].imshow(ground_truth, cmap=cmap, vmin=0, vmax=num_classes)
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Plot Predicted Labels
    axes[1].imshow(predicted_labels, cmap=cmap, vmin=0, vmax=num_classes)
    axes[1].set_title(title2)
    axes[1].axis("off")

    # Show plots
    plt.tight_layout()
    plt.show()


def plot_loss(feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot([it.feature_extractor_loss for it in feedback], label="loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_f1_score(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], size=(12, 6)
):
    plt.figure(figsize=size)
    plt.plot([it.metrics.f1_score for it in feedback], label="F1-score")
    plt.title("Eval F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()


def plot_overall_accuracy(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], size=(12, 6)
):
    plt.figure(figsize=size)
    plt.plot([it.metrics.overall_accuracy for it in feedback], label="Accuracy")
    plt.title("Eval Overall Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Overall Accuracy")
    plt.legend()
    plt.show()


def plot_average_accuracy(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], size=(12, 6)
):
    plt.figure(figsize=size)
    plt.plot([it.metrics.average_accuracy for it in feedback], label="Accuracy")
    plt.title("Eval Average Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.legend()
    plt.show()


def plot_kappa(feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot([it.metrics.kappa_score for it in feedback], label="Kappa")
    plt.title("Eval kappa score")
    plt.xlabel("Epoch")
    plt.ylabel("Kappa")
    plt.legend()
    plt.show()


def plot_progress_animation(predictions, num_classes) -> animation.FuncAnimation:
    num_frames = len(predictions)
    fig, ax = plt.subplots()
    cmap = plt.colormaps.get_cmap("jet")
    img = ax.imshow(predictions[0], cmap=cmap, vmin=0, vmax=num_classes)

    def update(frame):
        img.set_array(predictions[frame])
        ax.set_title(f"#{frame}")
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=200, blit=True
    )

    return ani


def plot_k_values(k_values, k_star, num_classes, size=(8, 5)):
    plt.figure(figsize=size)
    plt.plot(k_values, marker="o", linestyle="-", color="b", label="Number of Clusters")
    plt.axhline(y=k_star, color="r", linestyle="--", label="Target K*")
    plt.axhline(y=num_classes, color="g", linestyle="--", label="Number of classes")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Clusters")
    plt.title("Decrease in Number of Clusters")
    plt.legend()
    plt.grid()
    plt.show()


def plot_by_split_progress(splits, num_classes) -> animation.FuncAnimation:
    num_frames = len(splits)
    num_splits = len(splits[0])
    cmap = plt.colormaps.get_cmap("jet")

    fig, axes = plt.subplots(1, num_splits, figsize=(10, 5))
    images = []

    if num_splits == 1:
        images.append(axes.imshow(splits[0][0], cmap=cmap, vmin=0, vmax=num_classes))
        axes.set_title(f"#{0}.{0}")
        axes.axis("off")
    else:
        for i in range(num_splits):
            images.append(
                axes[i].imshow(splits[0][i], cmap=cmap, vmin=0, vmax=num_classes)
            )
            axes[i].set_title(f"#{0}.{i}")
            axes[i].axis("off")

    def update(frame):
        if num_splits == 1:
            images[0].set_array(splits[frame][0])
            axes.set_title(f"#{frame}.{0}")
        else:
            for i in range(num_splits):
                images[i].set_array(splits[frame][i])
                axes[i].set_title(f"#{frame}.{i}")

        return images

    return animation.FuncAnimation(
        fig, update, frames=num_frames, interval=200, blit=True
    )


def plot_extracted_features_by_epoch(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry],
    h: int,
    w: int,
    num_classes: int,
) -> animation.FuncAnimation:
    extracted_features = [
        [
            np.argmax(it, axis=1).reshape(h, w)
            for it in h_e.step_snapshots.extracted_features
        ]
        for h_e in feedback[1:]
    ]

    return plot_by_split_progress(extracted_features, num_classes)


def plot_clusters_by_epoch(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry],
    h: int,
    w: int,
) -> animation.FuncAnimation:
    semantic_constraint = [
        [it.reshape(h, w) for it in h_e.step_snapshots.clustering_result]
        for h_e in feedback
    ]

    return plot_by_split_progress(semantic_constraint, None)


def plot_semantic_constraints_by_epoch(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry],
    h: int,
    w: int,
    num_classes: int,
) -> animation.FuncAnimation:
    semantic_constraint = [
        [it.reshape(h, w) for it in h_e.step_snapshots.semantic_constraint]
        for h_e in feedback
    ]

    return plot_by_split_progress(semantic_constraint, num_classes)


def plot_merged_semantic_constraint_by_epoch(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry],
    h: int,
    w: int,
    num_classes: int,
) -> animation.FuncAnimation:
    prediction_attempts = [
        it.step_snapshots.merged_semantic_constraint.reshape(h, w) for it in feedback
    ]
    return plot_progress_animation(prediction_attempts, num_classes)


def plot_predictions_by_epoch(
    feedback: list[SpatialRegulatedSelfTrainingHistoryEntry], num_classes: int
) -> animation.FuncAnimation:
    prediction_attempts = [
        it.step_snapshots.spatial_constraint_result for it in feedback
    ]
    return plot_progress_animation(prediction_attempts, num_classes)
