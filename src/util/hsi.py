from enum import Enum
from pathlib import Path
import numpy as np
import torch

from torch import nn
from torch.utils import data
from sklearn.decomposition import NMF, PCA, FactorAnalysis, TruncatedSVD
from sklearn.discriminant_analysis import StandardScaler

from src.definitions import CACHE_FOLDER
from src.model.autoencoder import (
    AsymmetricPointWiseAutoEncoder,
    SpatialAutoEncoder,
    SymmetricPointWiseAutoEncoder,
)
from src.trainer.autoencoder_trainer import AutoEncoderTrainer
from src.trainer.base_trainer import AdamOptimizedModule, TrainableModule

MAX_ITER = 10_000_000


class PreProcessType(Enum):
    STANDARTIZATION = "STANDARTIZATION"
    NORMALIZATION = "NORMALIZATION"
    NOPE = "NOPE"


class DimReductionType(Enum):
    PCA = "PCA"
    FA = "FA"
    SVD = "SVD"
    NMF = "NMF"
    SPARTIAL_AUTOENCODER = "SPARTIAL_AUTOENCODER"
    SYMETRIC_POINTWISE_AUTOENCODER = "SYMETRIC_POINTWISE_AUTOENCODER"
    ASYMETRIC_POINTWISE_AUTOENCODER = "ASYMETRIC_POINTWISE_AUTOENCODER"
    NOPE = "NOPE"


def reduce_hsi_dim(
    image: np.ndarray,
    out_dim: int,
    alg: DimReductionType,
    device: torch.device,
    random_state: int = 42,
    auto_encoder_epochs: int = 100,
    auto_encoder_lr: float = 1e-3,
) -> np.array:
    _, _, c = image.shape

    if c == out_dim or alg == DimReductionType.NOPE:
        return None, c, image
    elif alg == DimReductionType.PCA:
        return reduce_depth_with_pca(image, out_dim, random_state)
    elif alg == DimReductionType.FA:
        return reduce_depth_with_fa(image, out_dim, random_state)
    elif alg == DimReductionType.SVD:
        return reduce_depth_with_svd(image, out_dim, random_state)
    elif alg == DimReductionType.NMF:
        return reduce_depth_with_nmf(image, out_dim, random_state)
    elif alg == DimReductionType.SPARTIAL_AUTOENCODER:
        model = AdamOptimizedModule(
            SpatialAutoEncoder(input_channels=c, embedding_size=out_dim),
            lr=auto_encoder_lr,
        )
        return (
            model,
            out_dim,
            reduce_depth_with_patched_autoencoder(
                image=image,
                patch_size=9,
                model=model,
                trainer=AutoEncoderTrainer(nn.MSELoss(), auto_encoder_epochs, device),
                batch_size=64,
            ),
        )
    elif alg == DimReductionType.SYMETRIC_POINTWISE_AUTOENCODER:
        model = AdamOptimizedModule(
            SymmetricPointWiseAutoEncoder(units=[c, 150, 100, out_dim]),
            lr=auto_encoder_lr,
        )
        return (
            model,
            out_dim,
            reduce_depth_with_patched_autoencoder(
                image=image,
                patch_size=9,
                model=model,
                trainer=AutoEncoderTrainer(nn.MSELoss(), auto_encoder_epochs, device),
                batch_size=64,
            ),
        )
    elif alg == DimReductionType.ASYMETRIC_POINTWISE_AUTOENCODER:
        model = AdamOptimizedModule(
            AsymmetricPointWiseAutoEncoder(
                encoder_units_def=[c, 150, 100, out_dim],
                decoder_units_def=[out_dim, c],
            ),
            lr=auto_encoder_lr,
        )
        return (
            model,
            out_dim,
            reduce_depth_with_patched_autoencoder(
                image=image,
                patch_size=9,
                model=model,
                trainer=AutoEncoderTrainer(nn.MSELoss(), auto_encoder_epochs, device),
                batch_size=64,
            ),
        )


def preprocess_hsi(image: np.ndarray, alg: PreProcessType) -> tuple[object, np.ndarray]:
    if alg == PreProcessType.NOPE:
        return None, image
    elif alg == PreProcessType.STANDARTIZATION:
        return scale_image(image)
    elif alg == PreProcessType.NORMALIZATION:
        return None, normalize_hsi(image)


def value_counts_array(arr, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for val in arr:
        if val != 0:
            counts[val - 1] += 1
    return counts


def value_counts_array_with_zeros(arr, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for val in arr:
        counts[val] += 1
    return counts


def sample_fraction_from_segmentation(source, fraction_of_examples):
    source_arr = source.reshape(-1)
    result = sample_fraction_from_segmentation_vector(source_arr, fraction_of_examples)

    return result.reshape(source.shape)


def sample_fraction_from_segmentation_vector(source, fraction_of_examples):
    if fraction_of_examples == 1:
        return source

    len_source = len(source)
    num_classes = len(np.unique(source)) - 1
    examples_per_class = (
        value_counts_array(source, num_classes) * fraction_of_examples
    ).astype(int)
    examples_count = np.zeros(num_classes)
    result = np.zeros(source.shape)
    iter_count = 0

    while np.all(examples_count < examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source[i]
        examples_count_i = it - 1

        if (
            it > 0
            and examples_count[examples_count_i] < examples_per_class[examples_count_i]
            and result[i] == 0
        ):
            examples_count[examples_count_i] += 1
            result[i] = it

        iter_count += 1

    return result


def sample_fraction_from_segmentation_vector_with_zeros(source, fraction_of_examples):
    if fraction_of_examples == 1:
        return source

    len_source = len(source)
    num_classes = len(np.unique(source))
    examples_per_class = (
        value_counts_array_with_zeros(source, num_classes) * fraction_of_examples
    ).astype(int)
    examples_count = np.zeros(num_classes)
    result = np.full(source.shape, -1)
    iter_count = 0

    while np.all(examples_count < examples_per_class):
        if iter_count > MAX_ITER / 20:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source[i]
        examples_count_i = it

        if (
            examples_count[examples_count_i] < examples_per_class[examples_count_i]
            and result[i] == -1
        ):
            examples_count[examples_count_i] += 1
            result[i] = it

        iter_count += 1

    return result


def sample_from_segmentation_matrix(source, examples_per_class):
    source_arr = source.reshape(-1)
    len_source = len(source_arr)
    num_classes = len(np.unique(source_arr)) - 1
    examples_count = np.zeros(num_classes)
    result = np.zeros(len_source)
    iter_count = 0

    while not np.all(examples_count == examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source_arr[i]
        examples_count_i = it - 1

        if it > 0 and examples_count[examples_count_i] < examples_per_class:
            examples_count[examples_count_i] = examples_count[examples_count_i] + 1
            result[i] = it

        iter_count += 1

    return result.reshape(source.shape)


def mask_patched_fraction(x, y, fraction_of_examples, device):
    y_masked = sample_fraction_from_segmentation_vector_with_zeros(
        y, fraction_of_examples
    )
    mask = y_masked > -1

    x_labeled = x[mask, :, :, :]
    y_labeled = y[mask]
    x_unlabeled = x[~mask, :, :, :]
    y_unlabeled = y[~mask]

    x_full_tensor = torch.tensor(x, dtype=torch.float32, device=device).permute(
        0, 3, 1, 2
    )
    y_full_tensor = torch.tensor(y, dtype=torch.long, device=device)
    x_labeled_tensor = torch.tensor(
        x_labeled, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long, device=device)
    x_unlabeled_tensor = torch.tensor(
        x_unlabeled, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    y_unlabeled_tensor = torch.tensor(y_unlabeled, dtype=torch.long, device=device)

    return (
        data.TensorDataset(x_full_tensor, y_full_tensor),
        data.TensorDataset(x_labeled_tensor, y_labeled_tensor),
        data.TensorDataset(x_unlabeled_tensor, y_unlabeled_tensor),
        y_masked,
    )


def sample_from_segmentation_matrix_with_zeros(
    source, examples_per_class, cache_folder: Path
):
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_name = f"mask_{"".join([str(it) for it in examples_per_class])}"
    cache_path = cache_folder / cache_name

    if cache_path.exists():
        return np.load(cache_path)

    len_source = len(source)
    num_classes = len(np.unique(source))
    examples_count = np.zeros(num_classes)
    result = np.full(len_source, -1)
    iter_count = 0

    while not np.all(examples_count == examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError(
                f"Max number of iterations exceeded. Examples count: {examples_count}"
            )

        i = np.random.randint(low=0, high=len_source)
        it = source[i]
        examples_count_i = it

        if (
            result[i] == -1
            and examples_count[examples_count_i] < examples_per_class[examples_count_i]
        ):
            examples_count[examples_count_i] = examples_count[examples_count_i] + 1
            result[i] = it

        iter_count += 1

    np.save(cache_path, result)

    return result


def scale_image(image):
    h, w, c = image.shape
    img_reshaped = image.reshape(-1, c)
    scaler = StandardScaler()
    img_scaled = scaler.fit_transform(img_reshaped)

    return scaler, img_scaled.reshape(h, w, c)


def scale_patched(x):
    _, h, w, _ = x.shape
    scaler = StandardScaler()
    x = x.reshape(-1, x.shape[-1])
    x = scaler.fit_transform(x)
    x = x.reshape(-1, h, w, x.shape[-1])
    return scaler, x


def normalize_hsi(input):
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:, :, i])
        input_min = np.min(input[:, :, i])
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / (
            input_max - input_min
        )

    return input_normalize


def mirror_hsi(input_normalize, patch=5):
    height, width, band = input_normalize.shape

    padding = patch // 2
    mirror_hsi = np.zeros(
        (height + 2 * padding, width + 2 * padding, band), dtype=float
    )

    mirror_hsi[padding : (padding + height), padding : (padding + width), :] = (
        input_normalize
    )

    for i in range(padding):
        mirror_hsi[padding : (height + padding), i, :] = input_normalize[
            :, padding - i - 1, :
        ]

    for i in range(padding):
        mirror_hsi[padding : (height + padding), width + padding + i, :] = (
            input_normalize[:, width - 1 - i, :]
        )

    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[
            height + padding - 1 - i, :, :
        ]

    return mirror_hsi


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros(
        (x_train.shape[0], patch * patch * band_patch, band), dtype=float
    )

    x_train_band[:, nn * patch * patch : (nn + 1) * patch * patch, :] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch : (i + 1) * patch * patch, : i + 1] = (
                x_train_reshape[:, :, band - i - 1 :]
            )
            x_train_band[:, i * patch * patch : (i + 1) * patch * patch, i + 1 :] = (
                x_train_reshape[:, :, : band - i - 1]
            )
        else:
            x_train_band[:, i : (i + 1), : (nn - i)] = x_train_reshape[
                :, 0:1, (band - nn + i) :
            ]
            x_train_band[:, i : (i + 1), (nn - i) :] = x_train_reshape[
                :, 0:1, : (band - nn + i)
            ]

    for i in range(nn):
        if pp > 0:
            x_train_band[
                :,
                (nn + i + 1) * patch * patch : (nn + i + 2) * patch * patch,
                : band - i - 1,
            ] = x_train_reshape[:, :, i + 1 :]
            x_train_band[
                :,
                (nn + i + 1) * patch * patch : (nn + i + 2) * patch * patch,
                band - i - 1 :,
            ] = x_train_reshape[:, :, : i + 1]
        else:
            x_train_band[:, (nn + 1 + i) : (nn + 2 + i), (band - i - 1) :] = (
                x_train_reshape[:, 0:1, : (i + 1)]
            )
            x_train_band[:, (nn + 1 + i) : (nn + 2 + i), : (band - i - 1)] = (
                x_train_reshape[:, 0:1, (i + 1) :]
            )
    return x_train_band


def reduce_depth_with_autoencoder(image, model, trainer, device):
    autoencoder_x = (
        torch.tensor(image, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
    )

    train_autoencoder_loader = data.DataLoader(
        data.TensorDataset(autoencoder_x, torch.zeros(autoencoder_x.shape[0]))
    )

    trainer.fit(model, train_autoencoder_loader)

    encoded, _ = model(autoencoder_x)

    return encoded.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def reduce_depth_with_patched_autoencoder(
    image,
    patch_size: int,
    model: TrainableModule,
    trainer: AutoEncoderTrainer,
    batch_size: int = 128,
):
    h, w, _ = image.shape
    labels = np.zeros((h, w))

    x, y = extract_patches(image, labels, patch_size=patch_size)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=trainer.device).permute(
        0, 3, 1, 2
    )
    y_tensor = torch.tensor(y, dtype=torch.long, device=trainer.device)

    train_loader = data.DataLoader(
        data.TensorDataset(x_tensor, y_tensor), batch_size=batch_size
    )

    trainer.fit(model, train_loader)

    encoded, _ = model(x_tensor)

    return encoded.reshape(h, w, -1).detach().cpu().numpy()


def reduce_depth_with_pca(input, n_components, random_state=42):
    new_x = np.reshape(input, (-1, input.shape[2]))
    alg = PCA(n_components=n_components, whiten=True, random_state=random_state)
    new_x = alg.fit_transform(new_x)
    new_x = np.reshape(new_x, (input.shape[0], input.shape[1], n_components))
    return alg, n_components, new_x


def reduce_depth_with_fa(input, n_components, random_state=42):
    new_x = np.reshape(input, (-1, input.shape[2]))
    alg = FactorAnalysis(n_components=n_components, random_state=random_state)
    new_x = alg.fit_transform(new_x)
    new_x = np.reshape(new_x, (input.shape[0], input.shape[1], n_components))
    return alg, n_components, new_x


def reduce_depth_with_svd(input, n_components, random_state=42):
    new_x = np.reshape(input, (-1, input.shape[2]))
    alg = TruncatedSVD(n_components=n_components, random_state=random_state)
    new_x = alg.fit_transform(new_x)
    new_x = np.reshape(new_x, (input.shape[0], input.shape[1], n_components))
    return alg, n_components, new_x


def reduce_depth_with_nmf(input, n_components, random_state=42):
    new_x = np.reshape(input, (-1, input.shape[2]))
    alg = NMF(n_components=n_components, random_state=random_state)
    new_x = alg.fit_transform(new_x)
    new_x = np.reshape(new_x, (input.shape[0], input.shape[1], n_components))
    return alg, n_components, new_x


def extract_patches(image, labels, patch_size=5):
    return extract_image_patches(image, patch_size), extract_label_patches(labels)


def extract_label_patches(labels):
    return labels.ravel()


def extract_image_patches(image, patch_size=5):
    padded_image = mirror_hsi(image, patch_size)

    x = []
    h, w, _ = image.shape

    for i in range(h):
        for j in range(w):
            patch = padded_image[i : i + patch_size, j : j + patch_size, :]
            x.append(patch)

    return np.array(x)


def extract_band_patches(x, band_patch=3):
    _, patch, patch, band = x.shape

    return gain_neighborhood_band(x, band, band_patch, patch)


def train_test_band_patch_split(
    x, y, examples_per_class, cache_key, cache_folder=CACHE_FOLDER / "y_masked"
):
    y_masked = sample_from_segmentation_matrix_with_zeros(
        y, examples_per_class, cache_folder / cache_key
    )
    mask = y_masked > -1

    x_train = x[mask, :, :]
    y_train = y[mask]
    x_test = x[~mask, :, :]
    y_test = y[~mask]

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        y_masked,
    )


def slice_and_patch(image, patch_size=5, splits=4):
    splitted = np.array_split(image, splits, axis=2)
    return [extract_image_patches(it, patch_size) for it in splitted]
