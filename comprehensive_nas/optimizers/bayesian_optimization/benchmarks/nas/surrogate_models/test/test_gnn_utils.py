import unittest

import numpy as np
from surrogate_models.gnn.gnn_utils import NASBenchDataset


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.nasbench_dataset = NASBenchDataset(
            "surrogate_models/test/",
            result_paths=["surrogate_models/test/results_fidelity_0/results_0.json"],
            config_space_path="configspace301.json",
        )

    def test_length(self):
        self.assertEqual(1, len(self.nasbench_dataset))

    def test_correct_adjacency_matrix(self):
        (
            config_space_instance,
            val_accuracy,
            test_accuracy,
            json_file,
        ) = self.nasbench_dataset.config_loader[
            "surrogate_models/test/results_fidelity_0/results_0.json"
        ]
        (
            normal_cell,
            reduction_cell,
        ) = self.nasbench_dataset.create_darts_adjacency_matrix_from_config(
            config_space_instance
        )
        gt_normal_adjacency_matrix = np.array(
            [
                [0, 0, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self.assertTrue((normal_cell[0] == gt_normal_adjacency_matrix).all())

    def test_correct_coo_format(self):
        (
            config_space_instance,
            val_accuracy,
            test_accuracy,
            json_file,
        ) = self.nasbench_dataset.config_loader[
            "surrogate_models/test/results_fidelity_0/results_0.json"
        ]
        normal_cell, _ = self.nasbench_dataset.create_darts_adjacency_matrix_from_config(
            config_space_instance
        )
        normal_cell_pt = self.nasbench_dataset.convert_to_pytorch_format(normal_cell)

        x = [
            0.0,
            7.0,
            0.0,
            8.0,
            0.0,
            9.0,
            0.0,
            10.0,
            1.0,
            11.0,
            1.0,
            12.0,
            2.0,
            13.0,
            3.0,
            14.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
        y = [
            7.0,
            2.0,
            8.0,
            3.0,
            9.0,
            4.0,
            10.0,
            5.0,
            11.0,
            2.0,
            12.0,
            5.0,
            13.0,
            3.0,
            14.0,
            4.0,
            6.0,
            6.0,
            6.0,
            6.0,
        ]
        expected_coo_format = np.array([x, y], dtype=np.float64)

        self.assertTrue((normal_cell_pt[0] == expected_coo_format).all())


if __name__ == "__main__":
    unittest.main()
