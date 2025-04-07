import unittest

import numpy as np
import torch

from src.trainer.spatial_regulated_self_trainer import (
    SpatialRegulatedSelfTrainer,
    SpatialRegulatedSelfTrainerArgs,
)


class SpatialRegulatedSelfTrainerTest(unittest.TestCase):

    def setUp(self):
        """Initialize the class with mock args"""
        self.device = torch.device("cpu")

    def test_introduce_semantic_constraint_basic_case(self):
        """cluster 1 has more then threshold label 2"""
        args = SpatialRegulatedSelfTrainerArgs(
            model=None,
            optimizer=None,
            num_classes=3,
            over_cluster_count=6,
            over_cluster_count_decay=1,
            semantic_threshold=0.5,
        )
        test_instance = SpatialRegulatedSelfTrainer(args, self.device)
        cluster = [0, 0, 1, 1, 1, 1, 1, 2, 2]
        labels = [0, 0, 1, 1, 2, 2, 2, 0, 0]

        expected_output = np.array([0, 0, 2, 2, 2, 2, 2, 0, 0])

        result = test_instance.introduce_semantic_constraint(cluster, labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_introduce_semantic_constraint_less_then_threshold(self):
        """cluster 1 has less then threshold label 2"""
        args = SpatialRegulatedSelfTrainerArgs(
            model=None,
            optimizer=None,
            num_classes=3,
            over_cluster_count=6,
            over_cluster_count_decay=1,
            semantic_threshold=0.5,
        )
        test_instance = SpatialRegulatedSelfTrainer(args, self.device)
        cluster = [0, 0, 1, 1, 1, 1, 1, 2, 2]
        labels = [0, 0, 1, 2, 3, 4, 2, 0, 0]

        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = test_instance.introduce_semantic_constraint(cluster, labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_calculate_pseudo_label(self):
        """Test a case from the paper"""
        args = SpatialRegulatedSelfTrainerArgs(
            model=None,
            optimizer=None,
            num_classes=4,
            over_cluster_count=6,
            over_cluster_count_decay=1,
            spatial_threshold=8,
            spatial_constraint_weights=[1, 0.5],
        )
        test_instance = SpatialRegulatedSelfTrainer(args, self.device)

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


if __name__ == "__main__":
    unittest.main()
