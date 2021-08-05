from abc import abstractmethod
from typing import Iterable, Union

try:
    import torch
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)


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
