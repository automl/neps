from abc import abstractmethod
from typing import NoReturn


class AcquisitionOptimizer:
    def __init__(self, objective):
        self.objective = objective
        self.x: list = []
        self.y: list = []

    @abstractmethod
    def sample(self, pool_size: int) -> list:
        raise NotImplementedError

    def reset_XY(self, x: list, y: list) -> NoReturn:
        self.x = x
        self.y = y
