from copy import deepcopy

import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark

N_CATEGORICAL = 5
N_CONTINUOUS = 5


def evaluate_counting_ones(config, mode="eval", **kwargs):
    """(n_cont + n_cat)d CountingOnes test function
        input bounds:  0 <= xi <= 1, i = 1..n_cont
                            xj in [0, 1], j = 1..n_cat
        global optimum: [1] * (n_cont + n_cat),
        min function value = -1 * (n_cont + n_cat)
    """
    x = np.array(config.hps, dtype=float)
    y = float(np.sum(x))

    if mode == "test":
        return y
    else:
        return y, 1.0


class CountingOnes(AbstractBenchmark):
    def __init__(
        self,
        seed=None,
        optimize_arch=False,
        optimize_hps=True,
    ):
        super().__init__(seed, optimize_arch, optimize_hps)
        self.has_continuous_hp = bool(N_CONTINUOUS)
        self.has_categorical_hp = bool(N_CATEGORICAL)

    def sample_random_architecture(self):
        cs = CountingOnes.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(map(str, config.get_dictionary().values()))[:N_CATEGORICAL]
        rand_hps += list(config.get_dictionary().values())[N_CATEGORICAL:]
        self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init
        self.name = str(self.parse())

    def reinitialize(self, seed=None):
        self.seed = seed  # pylint: disable=attribute-defined-outside-init

    @staticmethod
    def get_config_space():
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
            "optima": ([1] * (N_CONTINUOUS + N_CATEGORICAL)),
            "bounds": [[0, 1] * N_CONTINUOUS, [0, 1] * N_CATEGORICAL],
            "f_opt": -10.0,
            "noise_variance": 0.05,
        }
