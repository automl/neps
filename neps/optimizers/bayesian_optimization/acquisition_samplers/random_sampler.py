from __future__ import annotations

from ....search_spaces.search_space import SearchSpace
from .base_acq_sampler import AcquisitionSampler


class RandomSampler(AcquisitionSampler):
    def __init__(
        self,
        patience: int = 100,
    ):
        super().__init__(patience=patience)

    def sample(self, acquisition_function=None) -> list[SearchSpace]:
        rand_config = self.search_space.copy().sample(patience=self.patience)
        return rand_config
