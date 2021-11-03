import time

import numpy as np

from comprehensive_nas.evaluation.objective import Objective


class Hartmann6(Objective):
    def __init__(self, seed=None, log_scale=False, negative=False):
        super().__init__(seed, log_scale, negative)

    def __call__(self, config, **kwargs):
        """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
        """

        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array(
            [
                [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
            ]
        )
        P = 0.0001 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        x = config.get_hps()

        start = time.time()

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2

            y += alpha[i] * np.exp(-internal_sum)

        end = time.time()

        return {
            "loss": -y,
            "info_dict": {
                "config_id": config.id,
                "val_score": -y,
                "test_score": -y,
                "train_time": end - start,
            },
        }
