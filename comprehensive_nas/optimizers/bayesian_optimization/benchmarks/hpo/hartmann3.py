import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark


class Hartmann3(AbstractBenchmark):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array(
        [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    )
    P = 0.0001 * np.array(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1090, 8732, 5547], [381, 5743, 8828]]
    )

    def __init__(
        self,
        log_scale=False,
        negative=False,
        seed=None,
        optimize_arch=False,
        optimize_hps=True,
    ):
        super().__init__(seed, negative, log_scale, optimize_arch, optimize_hps)
        self.has_continuous_hp = True
        self.has_categorical_hp = False

    def query(self, mode='eval', **kwargs):  # pylint: disable=unused-argument

        x = self.hps

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2

            y += self.alpha[i] * np.exp(-internal_sum)

        if self.negative:
            y = -y

        if mode == "test":
            return y
        else:
            return y, {"train_time": self.eval_cost()}

    @staticmethod
    def eval_cost():
        return 1.0

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative  # pylint: disable=attribute-defined-outside-init
        self.seed = seed  # pylint: disable=attribute-defined-outside-init

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Hartmann3.get_meta_information()["bounds"])
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "Hartmann3",
            "capital": 100,
            "optima": ([[0.114614, 0.555649, 0.852547]]),
            "bounds": [[0.0, 1.0]] * 3,
            "f_opt": -3.8627795317627736,
            "noise_variance": 0.01,
        }

    def sample_random_architecture(self):
        cs = Hartmann3.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(config.get_dictionary().values())
        self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init
        self.name = str(self.parse())
