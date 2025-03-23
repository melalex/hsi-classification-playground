import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt


def plot_segmentation_results(
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
