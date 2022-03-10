from __future__ import annotations

from abc import abstractmethod

import torch

from ....search_spaces.search_space import SearchSpace


class AcquisitionSampler:
    def __init__(self, patience: int = 50):
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.search_space: SearchSpace = None
        self.acquisition_function = None
        self.x: list = []
        self.y: list = []
        self.patience = patience

    @abstractmethod
    def sample(self, acquisition_function) -> SearchSpace:
        raise NotImplementedError

    def sample_batch(self, acquisition_function, batch) -> list[SearchSpace]:
        return [self.sample(acquisition_function) for _ in range(batch)]

    def work_with(self, search_space, x: list, y: list | torch.Tensor) -> None:
        self.search_space = search_space
        self.x = x
        self.y = y
