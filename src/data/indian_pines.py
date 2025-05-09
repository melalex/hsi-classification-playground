import scipy.io
import torch

from torch.utils.data import TensorDataset

from src.definitions import EXTERNAL_DATA_FOLDER
from src.util.image import scale_image
from src.util.kaggle import download_and_unzip
from src.util.patches import extract_patches, scale_patched


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
