from abc import ABCMeta

import numpy as np
import ConfigSpace as CS

from nasbowl.benchmarks.abstract_benchmark import AbstractBenchmark
from nasbowl.utils.nasbowl_utils import config_to_array


class Hartmann3_4(AbstractBenchmark):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1090, 8732, 5547],
                           [381, 5743, 8828]])

    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__()
        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.negative = negative
        self.log_scale = log_scale

    def eval(self, Gs, hps, **kwargs):

        x = hps
        fidelity = config_to_array(kwargs['fidelity'])

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2

            alpha_fidel = 0.1 * (1 - fidelity[i])  # according to the paper BOCA, see pg 18
            y += (self.alpha[i] - alpha_fidel) * np.exp(-internal_sum)

        if self.negative:
            y = -y

        return y, {'train_time': self.eval_cost(fidelity=fidelity)}

    def eval_cost(self, **kwargs):
        fidelity = config_to_array(kwargs['fidelity'])
        return 0.05 + (1 - 0.05) * fidelity[0] ** 3 * fidelity[1] ** 2 * fidelity[2] ** 1.5 * fidelity[3]

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    @staticmethod
    def get_config_space(**kwargs):
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
            Hartmann3_4.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_fidelity_space() -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
            Hartmann3_4.get_meta_information()['fidelity_bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Hartmann3_4',
                'capital': 100,
                'optima': ([[0.114614, 0.555649, 0.852547]]),
                'bounds': [[0.0, 1.0]] * 3,
                'fidelity_bounds': [[0.0, 1.0]] * 4,
                'f_opt': -3.8627795317627736,
                'noise_variance': 0.01}


class Hartmann3(Hartmann3_4):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__(multi_fidelity, log_scale, negative, seed)

    def eval(self, Gs, hps, **kwargs):
        return super(Hartmann3, self).eval(Gs, hps, fidelity=[1., 1., 1., 1.])

    def eval_cost(self, **kwargs):
        return 1.0

    @staticmethod
    def get_meta_information():
        meta = Hartmann3_4.get_meta_information()
        meta['name'] = 'Hartmann3'
        meta['fidelity_bounds'] = [[1.0, 1.0]] * 4
        return meta
