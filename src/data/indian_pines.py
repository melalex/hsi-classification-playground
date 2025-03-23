import scipy.io
from src.definitions import EXTERNAL_DATA_FOLDER
from src.util.kaggle import download_and_unzip


def load_indian_pines(dest=EXTERNAL_DATA_FOLDER):
    ds_path = download_and_unzip(
        "sciencelabwork", "hyperspectral-image-sensing-dataset-ground-truth", dest
    )
    
    x = scipy.io.loadmat(ds_path / "Indian_pines_corrected.mat")
    y = scipy.io.loadmat(ds_path / "Indian_pines_gt.mat")

    return x["indian_pines_corrected"], y["indian_pines_gt"]
