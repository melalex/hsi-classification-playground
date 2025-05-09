import numpy as np
import torch

import torch.nn.functional as F

from sklearn.discriminant_analysis import StandardScaler


def pad_image(image, margin):
    return np.pad(image, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")


def pad_tensor(image, margin):
    image = image.permute(2, 0, 1)
    padded = F.pad(image, pad=(margin, margin, margin, margin), mode="reflect")
    return padded.permute(1, 2, 0)


def extract_patches(image, labels, patch_size=5):
    return extract_image_patches(image, patch_size), extract_label_patches(labels)


def extract_label_patches(labels):
    return labels.ravel()


def extract_image_patches(image, patch_size=5):
    margin = patch_size // 2
    padded_image = pad_image(image, margin)

    x = []
    h, w, _ = image.shape

    for i in range(h):
        for j in range(w):
            patch = padded_image[i : i + patch_size, j : j + patch_size, :]
            x.append(patch)

    return np.array(x)


def extract_tensor_patches(image, patch_size=5):
    margin = patch_size // 2
    padded_image = pad_tensor(image, margin)

    x = []
    h, w, _ = image.shape

    for i in range(h):
        for j in range(w):
            patch = padded_image[i : i + patch_size, j : j + patch_size, :]
            x.append(patch)

    return torch.stack(x)


def slice_and_patch(image, patch_size=5, splits=4):
    splitted = np.array_split(image, splits, axis=2)
    return [extract_image_patches(it, patch_size) for it in splitted]


def scale_patched(x):
    _, h, w, _ = x.shape
    scaler = StandardScaler()
    x = x.reshape(-1, x.shape[-1])
    x = scaler.fit_transform(x)
    x = x.reshape(-1, h, w, x.shape[-1])
    return scaler, x
