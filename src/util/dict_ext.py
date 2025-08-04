from typing import Sequence


def arrange_and_repeat(times, value):
    return {it: value for it in range(times)}


def list_to_dict[A](arr: Sequence[A]) -> dict[int, A]:
    return {i: val for i, val in enumerate(arr)}
