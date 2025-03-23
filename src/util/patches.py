import numpy as np
from sklearn.discriminant_analysis import StandardScaler


def pad_image(image, margin):
    return np.pad(image, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")


def extract_patches(image, labels, patch_size=5, remove_background=False):
    margin = patch_size // 2
    padded_image = pad_image(image, margin)

    x = []
    y = []
    h, w = labels.shape

    for i in range(h):
        for j in range(w):
            if not remove_background or labels[i, j] > 0:
                patch = padded_image[i : i + patch_size, j : j + patch_size, :]
                x.append(patch)
                if remove_background:
                    y.append(labels[i, j] - 1)
                else:
                    y.append(labels[i, j])

    return np.array(x), np.array(y)


def scale_patched(x):
    _, h, w, _ = x.shape
    scaler = StandardScaler()
    x = x.reshape(-1, x.shape[-1])
    x = scaler.fit_transform(x)
    x = x.reshape(-1, h, w, x.shape[-1])
    return scaler, x
