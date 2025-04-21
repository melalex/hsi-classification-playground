import unittest

import numpy as np
import torch

from src.pipeline.common import (
    calculate_pseudo_label,
    introduce_semantic_constraint,
    merge_clustering_results,
)


class SpatialRegulatedSelfTrainingPipelineTest(unittest.TestCase):

    def setUp(self):
        """Initialize the class with mock args"""
        self.device = torch.device("cpu")

    def test_introduce_semantic_constraint_basic_case(self):
        """cluster 1 has more then threshold label 2"""
        num_classes = 3
        semantic_threshold = 0.5
        cluster = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2, 2, 0, 0])

        expected_output = np.array([0, 0, 2, 2, 2, 2, 2, 0, 0])

        result = introduce_semantic_constraint(
            cluster, labels, num_classes, semantic_threshold
        )
        np.testing.assert_array_equal(result, expected_output)

    def test_introduce_semantic_constraint_less_then_threshold(self):
        """cluster 1 has less then threshold label 2"""
        num_classes = 3
        semantic_threshold = 0.5
        cluster = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 2, 3, 4, 2, 0, 0])

        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = introduce_semantic_constraint(
            cluster, labels, num_classes, semantic_threshold
        )
        np.testing.assert_array_equal(result, expected_output)

    def test_calculate_pseudo_label(self):
        """Test a case from the paper"""
        spatial_threshold = 8
        spatial_constraint_weights = [1, 0.5]
        y = np.array(
            [
                [3, 2, 1, 0, 0],
                [0, 1, 1, 1, 1],
                [2, 2, 0, 1, 0],
                [0, 2, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ]
        )

        result = calculate_pseudo_label(
            2, 2, y, spatial_constraint_weights, spatial_threshold
        )

        np.testing.assert_equal(result, 1)

    def test_merge_clustering_results(self):
        """Test merge clustering result"""
        clustering_results = [
            np.array([1, 2, 1, 3, 2, 1, 2, 0, 3, 2]),
            np.array([1, 2, 1, 3, 2, 0, 2, 0, 1, 2]),
            np.array([1, 2, 1, 3, 2, 0, 2, 0, 1, 2]),
        ]

        actual = merge_clustering_results(clustering_results)

        np.testing.assert_equal(actual, np.array([1, 2, 1, 3, 2, 0, 2, 0, 0, 2]))


if __name__ == "__main__":
    unittest.main()
