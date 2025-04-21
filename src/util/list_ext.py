from collections import defaultdict


def group_indices(arr):
    index_groups = defaultdict(list)
    for index, value in enumerate(arr):
        index_groups[value].append(index)
    return dict(index_groups)


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
