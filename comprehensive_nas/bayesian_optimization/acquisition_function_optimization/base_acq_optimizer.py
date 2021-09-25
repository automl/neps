from abc import abstractmethod
from typing import Iterable, Union

try:
    import torch
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..acqusition_functions.base_acqusition import BaseAcquisition


class AcquisitionOptimizer:
    def __init__(self, objective, acquisition_function: BaseAcquisition):
        self.objective = objective
        self.acquisition_function = acquisition_function
        self.x: Iterable = []
        self.y: Iterable = []

    @abstractmethod
    def sample(self, pool_size: int) -> list:
        raise NotImplementedError

    def reset_surrogate_model(self, surrogate_model) -> None:
        self.acquisition_function.reset_surrogate_model(surrogate_model)

    def reset_XY(self, x: Iterable, y: Union[Iterable, torch.Tensor]) -> None:
        self.x = x
        self.y = y
