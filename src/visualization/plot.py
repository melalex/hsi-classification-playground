import numpy as np
import seaborn as sns

from matplotlib import animation, pyplot as plt


def plot_segmentation_comparison(
    ground_truth, predicted_labels, title1="Ground Truth", title2="Predicted"
):
    # Define color map (seaborn deep to get distinguishable colors)
    cmap = sns.color_palette("tab10", np.max(ground_truth) + 1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Ground Truth
    axes[0].imshow(ground_truth, cmap=plt.cm.get_cmap("jet", np.max(ground_truth) + 1))
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Plot Predicted Labels
    axes[1].imshow(
        predicted_labels, cmap=plt.cm.get_cmap("jet", np.max(predicted_labels) + 1)
    )
    axes[1].set_title(title2)
    axes[1].axis("off")

    # Show plots
    plt.tight_layout()
    plt.show()


def plot_loss(feedback, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot([it["loss"] for it in feedback], label="loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_f1_score(feedback, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot([it["val_f1"] for it in feedback], label="F1-score")
    plt.title("Eval F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()


def plot_accuracy(feedback, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot([it["val_accuracy"] for it in feedback], label="Accuracy")
    plt.title("Eval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_kappa(feedback, size=(12, 6)) :
    plt.figure(figsize=size)
    plt.plot([it["val_kappa"] for it in feedback], label="Kappa")
    plt.title("Eval kappa score")
    plt.xlabel("Epoch")
    plt.ylabel("Kappa")
    plt.legend()
    plt.show()

def plot_progress_animation(predictions):
    num_frames = len(predictions)
    fig, ax = plt.subplots()
    img = ax.imshow(predictions[0], cmap="jet", vmin=0, vmax=1)

    def update(frame):
        img.set_array(predictions[frame])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

    return ani
