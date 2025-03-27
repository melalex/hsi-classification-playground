from collections import defaultdict


def group_indices(arr):
    index_groups = defaultdict(list)
    for index, value in enumerate(arr):
        index_groups[value].append(index)
    return dict(index_groups)

