from abc import abstractmethod

import numpy as np


class Objective:
    def __init__(self, log_scale: bool, negative: bool = False) -> None:
        self.log_scale = log_scale
        self.negative = negative

    @abstractmethod
    def __call__(self, config, mode, **kwargs):
        raise NotImplementedError

    def transform(self, val):
        if self.log_scale:
            val = np.log(val + 1e-8)  # avoid log(0)
        if self.negative:
            val *= -1
        return val

    def inv_transform(self, val):
        if self.negative:
            val *= -1
        if self.log_scale:
            val = np.exp(val)
        return val
