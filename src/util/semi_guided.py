import numpy as np

MAX_ITER = 10_000_000


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
