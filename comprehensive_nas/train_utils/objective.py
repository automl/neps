from abc import abstractmethod

import numpy as np


class Objective:
    def __init__(self, log_scale: bool, negative: bool = False) -> None:
        self.log_scale = log_scale
        self.negative = negative

    @abstractmethod
    def __call__(self, config, mode, **kwargs):
        raise NotImplementedError

    def _transform_score(self, score):
        if self.log_scale:
            score = np.log(score + 1e-8)
        if self.negative:
            score = -score
        return score
