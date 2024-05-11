from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Sequence, Callable

from neps.utils.types import Array

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


class AcquisitionSampler:
    def __init__(self, pipeline_space: SearchSpace, patience: int = 50):
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.pipeline_space = pipeline_space
        self.acquisition_function = None
        self.x: list[SearchSpace] = []
        self.y: Sequence[float] | Array = []
        self.patience = patience

    @abstractmethod
    def sample(self, acquisition_function: Callable) -> SearchSpace:
        raise NotImplementedError

    def sample_batch(self, acquisition_function: Callable, batch: int) -> list[SearchSpace]:
        return [self.sample(acquisition_function) for _ in range(batch)]

    def set_state(self, x: list[SearchSpace], y: Sequence[float] | Array) -> None:
        self.x = x
        self.y = y
