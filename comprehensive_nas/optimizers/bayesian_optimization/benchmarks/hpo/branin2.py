import time

import numpy as np

from comprehensive_nas.evaluation.objective import Objective


class Branin2(Objective):
    def __init__(self, seed=None, log_scale=False, negative=False):
        super().__init__(seed, log_scale, negative)

    def __call__(self, config, **kwargs):
        """2d Branin test function
        input bounds:  -5 <= x1 <= 10,
                        0 <= x2 <= 15,
        global optima: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
        min function value = 0.39789
        """

        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        x = config.get_hps()

        start = time.time()
        y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
        y += s * (1 - t) * np.cos(x[0]) + s
        end = time.time()

        return {
            "loss": y,
            "info_dict": {
                "config_id": config.id,
                "val_score": y,
                "test_score": y,
                "train_time": end - start,
            },
        }
