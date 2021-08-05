from abc import abstractmethod
from typing import Iterable, Union

import torch


class AcquisitionOptimizer:
    def __init__(self, objective):
        self.objective = objective
        self.x: Iterable = []
        self.y: Iterable = []

    @abstractmethod
    def sample(self, pool_size: int) -> list:
        raise NotImplementedError

    def reset_XY(self, x: Iterable, y: Union[Iterable, torch.Tensor]) -> None:
        self.x = x
        self.y = y
