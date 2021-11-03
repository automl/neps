import time

import numpy as np

from comprehensive_nas.evaluation.objective import Objective


class Hartmann3(Objective):
    def __init__(self, seed=None, log_scale=False, negative=False):
        super().__init__(seed, log_scale, negative)

    def __call__(self, config, **kwargs):
        """3d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..3
        global optimum: (0.114614, 0.555649, 0.852547),
        min function value = -3.86278
        """

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array(
            [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
        )
        P = 0.0001 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1090, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        x = config.get_hps()

        start = time.time()

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
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
