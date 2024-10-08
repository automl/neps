from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


class BaseAcquisition(ABC):
    def __init__(self):
        self.surrogate_model: Any | None = None

    @abstractmethod
    def eval(
        self,
        x: Iterable,
        *,
        asscalar: bool = False,
    ) -> np.ndarray | torch.Tensor | float:
        """Evaluate the acquisition function at point x2."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray | torch.Tensor | float:
        return self.eval(*args, **kwargs)

    def set_state(self, surrogate_model: Any, **kwargs: Any) -> None:
        self.surrogate_model = surrogate_model
