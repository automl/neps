import ConfigSpace as CS
import numpy as np

from comprehensive_nas.bo.utils.nasbowl_utils import config_to_array

from ..abstract_benchmark import AbstractBenchmark


class Hartmann6_4(AbstractBenchmark):
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

    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__()
        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.negative = negative
        self.log_scale = log_scale

    def eval(self, Gs, hps, **kwargs):
        """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        fidelity bounds: 0 <= zi <= 1, i = 1..4
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
        """
        x = hps
        fidelity = config_to_array(kwargs["fidelity"])

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2

            alpha_fidel = 0.1 * (
                1 - fidelity[i]
            )  # according to the paper BOCA, see pg 18
            y += (self.alpha[i] - alpha_fidel) * np.exp(-internal_sum)

        if self.negative:
            y = -y

        return y, {"train_time": self.eval_cost(fidelity=fidelity)}

    @staticmethod
    def eval_cost(**kwargs):
        # according to the paper BOCA, see pg 18
        fidelity = config_to_array(kwargs["fidelity"])
        return (
            0.05
            + (1 - 0.05)
            * fidelity[0] ** 3
            * fidelity[1] ** 2
            * fidelity[2] ** 1.5
            * fidelity[3]
        )

    @staticmethod
    def get_config_space(**kwargs):
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
            Hartmann6_4.get_meta_information()["bounds"]
        )
        return cs

    @staticmethod
    def get_fidelity_space() -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
            Hartmann6_4.get_meta_information()["fidelity_bounds"]
        )
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "Hartmann6_4",
            "capital": 100,
            "optima": ([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
            "bounds": [[0.0, 1.0]] * 6,
            "fidelity_bounds": [[0.0, 1.0]] * 4,
            "f_opt": -3.322368011391339,
            "noise_variance": 0.05,
        }


class Hartmann6(Hartmann6_4):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__(multi_fidelity, log_scale, negative, seed)

    def eval(self, Gs, hps, **kwargs):
        return super().eval(Gs, hps, fidelity=[1.0, 1.0, 1.0, 1.0])

    def eval_cost(self, **kwargs):
        return 1.0

    @staticmethod
    def get_meta_information():
        meta = Hartmann6_4.get_meta_information()
        meta["name"] = "Hartmann6"
        meta["fidelity_bounds"] = [[1.0, 1.0]] * 4
        return meta
