import numpy as np


def exponential_decay_over_clustering(k_star, lambda_v, max_iter=12):
    result = k_star * np.exp(
        np.arange(max_iter, -1, -1) * lambda_v,
    )

    return result.astype(int)


def linear_over_clustering(max_cluster_size, k_star, max_iter=12):
    x1 = 0
    y1 = max_cluster_size
    x2 = max_iter
    y2 = k_star

    x_range = np.arange(x1, x2 + 1) if x1 < x2 else np.arange(x1, x2 - 1, -1)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    y_values = slope * x_range + intercept

    return y_values.astype(int)
