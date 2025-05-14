import numpy as np

MAX_ITER = 10_000_000


def value_counts_array(arr, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for val in arr:
        if val != 0:
            counts[val - 1] += 1
    return counts


def value_counts_array_with_zeros(arr, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for val in arr:
        counts[val] += 1
    return counts


def sample_fraction_from_segmentation(source, fraction_of_examples):
    source_arr = source.reshape(-1)
    result = sample_fraction_from_segmentation_vector(source_arr, fraction_of_examples)

    return result.reshape(source.shape)


def sample_fraction_from_segmentation_vector(source, fraction_of_examples):
    if fraction_of_examples == 1:
        return source

    len_source = len(source)
    num_classes = len(np.unique(source)) - 1
    examples_per_class = (
        value_counts_array(source, num_classes) * fraction_of_examples
    ).astype(int)
    examples_count = np.zeros(num_classes)
    result = np.zeros(source.shape)
    iter_count = 0

    while np.all(examples_count < examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source[i]
        examples_count_i = it - 1

        if (
            it > 0
            and examples_count[examples_count_i] < examples_per_class[examples_count_i]
            and result[i] == 0
        ):
            examples_count[examples_count_i] += 1
            result[i] = it

        iter_count += 1

    return result


def sample_fraction_from_segmentation_vector_with_zeros(source, fraction_of_examples):
    if fraction_of_examples == 1:
        return source

    len_source = len(source)
    num_classes = len(np.unique(source))
    examples_per_class = (
        value_counts_array_with_zeros(source, num_classes) * fraction_of_examples
    ).astype(int)
    examples_count = np.zeros(num_classes)
    result = np.full(source.shape, -1)
    iter_count = 0

    while np.all(examples_count < examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source[i]
        examples_count_i = it

        if (
            examples_count[examples_count_i] < examples_per_class[examples_count_i]
            and result[i] == -1
        ):
            examples_count[examples_count_i] += 1
            result[i] = it

        iter_count += 1

    return result


def sample_from_segmentation_matrix(source, examples_per_class):
    source_arr = source.reshape(-1)
    len_source = len(source_arr)
    num_classes = len(np.unique(source_arr)) - 1
    examples_count = np.zeros(num_classes)
    result = np.zeros(len_source)
    iter_count = 0

    while not np.all(examples_count == examples_per_class):
        if iter_count > MAX_ITER:
            raise RuntimeError("Max number of iterations exceeded")

        i = np.random.randint(low=0, high=len_source)
        it = source_arr[i]
        examples_count_i = it - 1

        if it > 0 and examples_count[examples_count_i] < examples_per_class:
            examples_count[examples_count_i] = examples_count[examples_count_i] + 1
            result[i] = it

        iter_count += 1

    return result.reshape(source.shape)
