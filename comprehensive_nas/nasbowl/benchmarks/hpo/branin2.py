import ConfigSpace
import numpy as np

import ConfigSpace as CS
from nasbowl.benchmarks.abstract_benchmark import AbstractBenchmark
from nasbowl.utils.nasbowl_utils import config_to_array


class Branin2_3(AbstractBenchmark):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__()
        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.negative = negative
        self.log_scale = log_scale

    def eval(self, Gs, hps, **kwargs):

        x = hps
        fidelity = config_to_array(kwargs['fidelity'])

        a = 1
        b = 5.1 / (4 * np.pi ** 2) - 0.01 * (1.0 - fidelity[0])
        c = 5 / np.pi - 0.1 * (1.0 - fidelity[1])
        r = 6
        s = 10
        t = 1 / (8 * np.pi) + 0.05 * (1.0 - fidelity[2])
        y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
        y += s * (1 - t) * np.cos(x[0]) + s

        if self.negative:
            y = -y
        return y, {'train_time': self.eval_cost(fidelity=fidelity)}

    def eval_cost(self, fidelity):
        fidelity = config_to_array(fidelity)
        return 0.05 + fidelity[0] ** 3 * fidelity[1] ** 2 * fidelity[2] ** 1.5

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    @staticmethod
    def get_config_space(**kwargs):
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Branin2_3.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_fidelity_space() -> ConfigSpace.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
            Branin2_3.get_meta_information()['fidelity_bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Branin2_3',
                'capital': 50,
                'optima': ([[-np.pi, 12.275],
                            [np.pi, 2.275],
                            [9.42478, 2.475]]),
                'bounds': [[-5, 10], [0, 15]],
                'fidelity_bounds': [[0.0, 1.0]] * 3,
                'f_opt': 0.39788735773,
                'noise_variance': 0.05}


class Branin2(Branin2_3):
    def __init__(self, multi_fidelity=False, log_scale=False, negative=False, seed=None):
        super().__init__(multi_fidelity, log_scale, negative, seed)

    def eval(self, Gs, hps, **kwargs):
        return super(Branin2, self).eval(Gs, hps, fidelity=[1., 1., 1.])

    def eval_cost(self, **kwargs):
        return 1.0

    @staticmethod
    def get_meta_information():
        meta = Branin2_3.get_meta_information()
        meta['name'] = 'Branin2'
        meta['fidelity_bounds'] = [[1.0, 1.0]] * 3
        return meta
