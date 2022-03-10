from abc import abstractmethod
from typing import Iterable, Union

import torch

from ..acquisition_functions.base_acquisition import BaseAcquisition


class AcquisitionOptimizer:
    def __init__(self, search_space, acquisition_function: BaseAcquisition = None):
        self.search_space = search_space
        self.acquisition_function = acquisition_function
        self.x: Iterable = []
        self.y: Iterable = []

    @abstractmethod
    def sample(self, pool_size: int, batch_size: int = None):
        raise NotImplementedError

    def reset_XY(self, x: Iterable, y: Union[Iterable, torch.Tensor]) -> None:
        self.x = x
        self.y = y
