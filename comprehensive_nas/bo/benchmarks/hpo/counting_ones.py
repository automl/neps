import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark

N_CATEGORICAL = 5
N_CONTINUOUS = 5


class CountingOnes(AbstractBenchmark):
    def __init__(
        self,
        log_scale=False,
        negative=False,
        seed=None,
        optimize_arch=False,
        optimize_hps=True,
    ):
        super().__init__(seed, negative, log_scale, optimize_arch, optimize_hps)

    def query(self):

        x = self.hps
        if isinstance(x[0], str):
            x[:N_CATEGORICAL] = list(map(int, x[:N_CATEGORICAL]))
        y = float(np.sum(x))

        if self.negative:
            y = -y
        return y, {"train_time": self.eval_cost()}

    @staticmethod
    def eval_cost():
        return 1.0

    def sample_random_architecture(self):
        cs = CountingOnes.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(map(str, config.get_dictionary().values()))[:N_CATEGORICAL]
        rand_hps += list(config.get_dictionary().values())[N_CATEGORICAL:]
        self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative  # pylint: disable=attribute-defined-outside-init
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
