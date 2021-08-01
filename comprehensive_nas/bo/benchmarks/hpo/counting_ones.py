import copy

import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark


N_CATEGORICAL = 5
N_CONTINUOUS = 5


class CountingOnes(AbstractBenchmark):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__()
        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.negative = negative
        self.log_scale = log_scale

    def eval(self, Gs, hps, **kwargs):

        x = copy.deepcopy(hps)
        if isinstance(x[0], str):
            x[:N_CATEGORICAL] = list(map(int, x[:N_CATEGORICAL]))
        y = float(np.sum(x))

        if self.negative:
            y = -y
        return y, {"train_time": self.eval_cost()}

    @staticmethod
    def eval_cost():
        return 1.0

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    @staticmethod
    def get_config_space(**kwargs):
        cs = CS.ConfigurationSpace()
        for i in range(N_CATEGORICAL):
            cs.add_hyperparameter(CS.CategoricalHyperparameter("cat_%d" % i, [0, 1]))
        for i in range(N_CONTINUOUS):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter("float_%d" % i, lower=0, upper=1)
            )
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "CountingOnes",
            "capital": 50,
            "optima": ([0]),
            "bounds": [[0, 1] * N_CONTINUOUS, [0, 1] * N_CATEGORICAL],
            "f_opt": -10.0,
            "noise_variance": 0.05,
        }
