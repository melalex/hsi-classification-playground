import scipy.io
import torch

from torch.utils.data import TensorDataset

from src.definitions import EXTERNAL_DATA_FOLDER
from src.util.image import scale_image
from src.util.kaggle import download_and_unzip
from src.util.patches import extract_patches, scale_patched
from src.util.semi_guided import sample_fraction_from_segmentation_vector_with_zeros


def load_indian_pines(dest=EXTERNAL_DATA_FOLDER):
    ds_path = download_and_unzip(
        "sciencelabwork", "hyperspectral-image-sensing-dataset-ground-truth", dest
    )

    x = scipy.io.loadmat(ds_path / "Indian_pines_corrected.mat")
    y = scipy.io.loadmat(ds_path / "Indian_pines_gt.mat")

    return x["indian_pines_corrected"], y["indian_pines_gt"]


def create_indian_pines_dataset(device, dest=EXTERNAL_DATA_FOLDER):
    x, y = load_indian_pines(dest)

    scaler, x = scale_image(x)

    x_tensor = (
        torch.tensor(x, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
    )
    y_tensor = torch.tensor(y, dtype=torch.long, device=device).unsqueeze(0)

    return scaler, TensorDataset(x_tensor, y_tensor)


def create_patched_indian_pines_dataset(
    device, patch_size=5, dest=EXTERNAL_DATA_FOLDER
):
    image, labels = load_indian_pines(dest)
    x, y = extract_patches(image, labels, patch_size=patch_size)
    scale, x = scale_patched(x)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return scale, TensorDataset(x_tensor, y_tensor)


# def create_patched_indian_pines_semi_guided_dataset(
#     patch_size=5, fraction_of_examples=0.1, dest=EXTERNAL_DATA_FOLDER
# ):
#     image, labels = load_indian_pines(dest)
#     x, y = extract_patches(image, labels, patch_size=patch_size)
#     scale, x = scale_patched(x)

#     full, labeled, unlabeled = mask_patched_indian_pines(x, y, fraction_of_examples)

#     return (scale, full, labeled, unlabeled)


# def mask_patched_indian_pines(x, y, fraction_of_examples, device):
#     y_masked = sample_fraction_from_segmentation_vector_with_zeros(
#         y, fraction_of_examples
#     )
#     mask = y_masked > -1

#     x_labeled = x[mask, :, :, :]
#     y_labeled = y[mask]
#     x_unlabeled = x[~mask, :, :, :]
#     y_unlabeled = y[~mask]

#     x_full_tensor = torch.tensor(x, dtype=torch.float32, device=device).permute(
#         0, 3, 1, 2
#     )
#     y_full_tensor = torch.tensor(y, dtype=torch.long, device=device)
#     x_labeled_tensor = torch.tensor(
#         x_labeled, dtype=torch.float32, device=device
#     ).permute(0, 3, 1, 2)
#     y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long, device=device)
#     x_unlabeled_tensor = torch.tensor(
#         x_unlabeled, dtype=torch.float32, device=device
#     ).permute(0, 3, 1, 2)
#     y_unlabeled_tensor = torch.tensor(y_unlabeled, dtype=torch.long, device=device)

#     return (
#         TensorDataset(x_full_tensor, y_full_tensor),
#         TensorDataset(x_labeled_tensor, y_labeled_tensor),
#         TensorDataset(x_unlabeled_tensor, y_unlabeled_tensor),
#     )
