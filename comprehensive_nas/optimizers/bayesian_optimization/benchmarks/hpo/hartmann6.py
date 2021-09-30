import ConfigSpace as CS
import numpy as np

from ..abstract_benchmark import AbstractBenchmark


class Hartmann6(AbstractBenchmark):
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

    def query(self, **kwargs):  # pylint: disable=unused-argument
        """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        fidelity bounds: 0 <= zi <= 1, i = 1..4
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
        """
        x = self.hps

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2

            y += self.alpha[i] * np.exp(-internal_sum)

        if self.negative:
            y = -y

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
        cs.generate_all_continuous_from_bounds(Hartmann6.get_meta_information()["bounds"])
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "Hartmann6",
            "capital": 100,
            "optima": ([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
            "bounds": [[0.0, 1.0]] * 6,
            "f_opt": -3.322368011391339,
            "noise_variance": 0.05,
        }

    def sample_random_architecture(self):
        cs = Hartmann6.get_config_space()
        config = cs.sample_configuration()
        rand_hps = list(config.get_dictionary().values())
        self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init
