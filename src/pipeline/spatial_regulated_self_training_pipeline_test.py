import unittest

import numpy as np
import torch

from src.pipeline.spatial_regulated_self_training_pipeline import (
    SpatialRegulatedSelfTrainingPipeline,
    SpatialRegulatedSelfTrainingPipelineArgs,
)


class SpatialRegulatedSelfTrainingPipelineTest(unittest.TestCase):

    def setUp(self):
        """Initialize the class with mock args"""
        self.device = torch.device("cpu")

    def test_introduce_semantic_constraint_basic_case(self):
        """cluster 1 has more then threshold label 2"""
        args = SpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=3,
            cluster_sizes=[6],
            feature_extractor=None,
            clustering=None,
            splits=1,
            patch_size=9,
            init_patch_size=5,
            semantic_threshold=0.5,
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
            record_step_snapshots=True,
        )
        test_instance = SpatialRegulatedSelfTrainingPipeline(args, self.device)
        cluster = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2, 2, 0, 0])

        expected_output = np.array([0, 0, 2, 2, 2, 2, 2, 0, 0])

        result = test_instance.introduce_semantic_constraint(cluster, labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_introduce_semantic_constraint_less_then_threshold(self):
        """cluster 1 has less then threshold label 2"""
        args = SpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=3,
            cluster_sizes=[6],
            feature_extractor=None,
            clustering=None,
            splits=1,
            patch_size=9,
            init_patch_size=5,
            semantic_threshold=0.5,
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
            record_step_snapshots=True,
        )
        test_instance = SpatialRegulatedSelfTrainingPipeline(args, self.device)
        cluster = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 2, 3, 4, 2, 0, 0])

        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = test_instance.introduce_semantic_constraint(cluster, labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_calculate_pseudo_label(self):
        """Test a case from the paper"""
        args = SpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=3,
            cluster_sizes=[6],
            feature_extractor=None,
            clustering=None,
            splits=1,
            patch_size=9,
            init_patch_size=5,
            semantic_threshold=0.5,
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
            record_step_snapshots=True,
        )
        test_instance = SpatialRegulatedSelfTrainingPipeline(args, self.device)

        y = np.array(
            [
                [3, 2, 1, 0, 0],
                [0, 1, 1, 1, 1],
                [2, 2, 0, 1, 0],
                [0, 2, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ]
        )

        result = test_instance.calculate_pseudo_label(2, 2, y)

        np.testing.assert_equal(result, 1)

    def test_merge_clustering_results(self):
        """Test merge clustering result"""
        args = SpatialRegulatedSelfTrainingPipelineArgs(
            num_classes=3,
            cluster_sizes=[6],
            feature_extractor=None,
            clustering=None,
            splits=1,
            patch_size=9,
            init_patch_size=5,
            semantic_threshold=0.5,
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
            record_step_snapshots=True,
        )
        test_instance = SpatialRegulatedSelfTrainingPipeline(args, self.device)

        clustering_results = [
            np.array([1, 2, 1, 3, 2, 1, 2, 0, 3, 2]),
            np.array([1, 2, 1, 3, 2, 0, 2, 0, 1, 2]),
            np.array([1, 2, 1, 3, 2, 0, 2, 0, 1, 2]),
        ]

        actual = test_instance.merge_clustering_results(clustering_results)

        np.testing.assert_equal(actual, np.array([1, 2, 1, 3, 2, 0, 2, 0, 0, 2]))


if __name__ == "__main__":
    unittest.main()
