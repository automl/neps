import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark

a = 1
b = 5.1 / (4 * np.pi ** 2)
c = 5 / np.pi
r = 6
s = 10
t = 1 / (8 * np.pi)


def evaluate_branin2(config, mode="eval", **kwargs):
    """2d Branin test function
    input bounds:  -5 <= x1 <= 10,
                    0 <= x2 <= 15,
    global optima: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    min function value = 0.39789
    """
    x = config.hps

    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
    y += s * (1 - t) * np.cos(x[0]) + s

    if mode == "test":
        return -y
    else:
        return -y, 1.0


class Branin2(AbstractBenchmark):
    def __init__(
        self,
        seed=None,
        optimize_arch=False,
        optimize_hps=True,
    ):
        super().__init__(seed, optimize_arch, optimize_hps)
        self.has_continuous_hp = True
        self.has_categorical_hp = False

    def reinitialize(self, seed=None):
        self.seed = seed  # pylint: disable=attribute-defined-outside-init

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Branin2.get_meta_information()["bounds"])
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "Branin2",
            "capital": 50,
            "optima": ([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]),
            "bounds": [[-5, 10], [0, 15]],
            "f_opt": 0.39788735773,
            "noise_variance": 0.05,
        }

    def sample(self):
        cs = Branin2.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(config.get_dictionary().values())
        self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init
        self.name = str(self.parse())
