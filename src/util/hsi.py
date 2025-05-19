import numpy as np
import torch
import torch.nn.functional as F

from torch.utils import data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from src.util.patches import extract_patches
from src.util.semi_guided import sample_from_segmentation_matrix_with_zeros


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


# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return (
        total_pos_train,
        total_pos_test,
        total_pos_true,
        number_train,
        number_test,
        number_true,
    )


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


# -------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x : (x + patch), y : (y + patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros(
        (x_train.shape[0], patch * patch * band_patch, band), dtype=float
    )
    # 中心区域
    x_train_band[:, nn * patch * patch : (nn + 1) * patch * patch, :] = x_train_reshape
    # 左边镜像
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
    # 右边镜像
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


# -------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(
    mirror_image, train_point, test_point, true_point, patch=5, band_patch=3
):
    _, _, band = mirror_image.shape
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(
            mirror_image, train_point, i, patch
        )
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print(
        "x_train_band shape = {}, type = {}".format(
            x_train_band.shape, x_train_band.dtype
        )
    )
    print(
        "x_test_band  shape = {}, type = {}".format(
            x_test_band.shape, x_test_band.dtype
        )
    )
    print(
        "x_true_band  shape = {}, type = {}".format(
            x_true_band.shape, x_true_band.dtype
        )
    )
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band


# -------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes + 1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true


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
    image, patch_size, model, trainers, device, batch_size=128
):
    h, w, _ = image.shape
    labels = np.zeros((h, w))

    x, y = extract_patches(image, labels, patch_size=patch_size)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    train_loader = data.DataLoader(
        data.TensorDataset(x_tensor, y_tensor), batch_size=batch_size
    )

    for trainer in trainers:
        trainer.fit(model, train_loader)

    encoded, _ = model(x_tensor)

    return encoded.reshape(h, w, -1).detach().cpu().numpy()


def reduce_depth_with_pca(input, n_components):
    h, w, c = input.shape
    reshaped_data = input.reshape(c, -1).T

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_data)

    reduced_image = reduced_data.T.reshape(h, w, n_components)

    return pca, reduced_image


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

def train_test_band_patch_split(x, y, examples_per_class):
    y_masked = sample_from_segmentation_matrix_with_zeros(y, examples_per_class)
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
