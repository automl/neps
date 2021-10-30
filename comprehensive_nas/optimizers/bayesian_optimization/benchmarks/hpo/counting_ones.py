import time

import numpy as np

from comprehensive_nas.evaluation.objective import Objective


class CountingOnes(Objective):
    def __init__(self, seed=None, log_scale=False, negative=False):
        super().__init__(seed, log_scale, negative)

    def __call__(self, config, **kwargs):
        """(n_cont + n_cat)d CountingOnes test function
        input bounds:  0 <= xi <= 1, i = 1..n_cont
                            xj in [0, 1], j = 1..n_cat
        global optimum: [1] * (n_cont + n_cat),
        min function value = -1 * (n_cont + n_cat)
        """

        x = np.array(config.get_hps(), dtype=float)
        start = time.time()
        y = -float(np.sum(x))
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
