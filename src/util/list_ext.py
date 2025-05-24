from collections import defaultdict

import numpy as np


def group_indices(arr):
    index_groups = defaultdict(list)
    for index, value in enumerate(arr):
        index_groups[value].append(index)
    return dict(index_groups)


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def smooth_moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def divide_interval(a: int, b: int, n: int) -> list[tuple[int,]]:
    if n <= 0:
        raise ValueError("Number of subintervals must be positive.")
    if a >= b:
        raise ValueError("Invalid interval: a must be less than b.")

    if n == 1:
        return [(a, b)]

    total_length = b - a + 1
    base_size = total_length // n
    remainder = total_length % n

    intervals = []
    start = a

    for i in range(n):
        end = start + base_size - 1
        if i < remainder:
            end += 1
        intervals.append((start, end))
        start = end + 1

    return intervals
