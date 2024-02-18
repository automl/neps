from __future__ import annotations

from abc import abstractmethod

import torch

from ....search_spaces.search_space import SearchSpace


class AcquisitionSampler:
    def __init__(self, pipeline_space: SearchSpace, patience: int = 50):
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.pipeline_space = pipeline_space
        self.acquisition_function = None
        self.x: list = []
        self.y: list = []
        self.patience = patience

    @abstractmethod
    def sample(self, acquisition_function) -> SearchSpace:
        raise NotImplementedError

    def sample_batch(self, acquisition_function, batch) -> list[SearchSpace]:
        return [self.sample(acquisition_function) for _ in range(batch)]

    def set_state(self, x: list, y: list | torch.Tensor) -> None:
        self.x = x
        self.y = y
