from typing import Sequence

import numpy as np


def arrange_and_repeat(times, value):
    return {it: value for it in range(times)}


def list_to_dict[A](arr: Sequence[A]) -> dict[int, A]:
    return {i: val for i, val in enumerate(arr)}


def unique_counts[A](arr: Sequence[A]) -> dict[A, int]:
    unique, counts = np.unique(arr, return_counts=True)

    return {u: c for u, c in zip(unique, counts)}
