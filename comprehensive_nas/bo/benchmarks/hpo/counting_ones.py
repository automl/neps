import copy

import ConfigSpace
import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark
from ..hyperconfiguration import HyperConfiguration


N_CATEGORICAL = 5
N_CONTINUOUS = 5


class CountingOnes(AbstractBenchmark):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__()
        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.negative = negative
        self.log_scale = log_scale

    def eval(self, config, **kwargs):

        x = copy.deepcopy(config.hps)
        if isinstance(x[0], str):
            x[:N_CATEGORICAL] = list(map(int, x[:N_CATEGORICAL]))
        y = float(np.sum(x))

        if self.negative:
            y = -y
        return y, {"train_time": self.eval_cost()}

    @staticmethod
    def eval_cost():
        return 1.0

    @staticmethod
    def sample(**kwargs):
        cs = CountingOnes.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(map(str, config.get_dictionary().values()))[:N_CATEGORICAL]
        rand_hps += list(config.get_dictionary().values())[N_CATEGORICAL:]
        return HyperConfiguration(graph=None, hps=rand_hps)

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
