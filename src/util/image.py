import numpy as np
from sklearn.decomposition import PCA
import torch

from torch.utils import data

from sklearn.discriminant_analysis import StandardScaler

from src.util.patches import extract_patches, scale_patched


def scale_image(image):
    h, w, c = image.shape
    img_reshaped = image.reshape(-1, c)
    scaler = StandardScaler()
    img_scaled = scaler.fit_transform(img_reshaped)

    return scaler, img_scaled.reshape(h, w, c)


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
