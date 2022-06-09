from __future__ import annotations

from ....search_spaces.search_space import SearchSpace
from .base_acq_sampler import AcquisitionSampler


class RandomSampler(AcquisitionSampler):
    def __init__(self, pipeline_space: SearchSpace, patience: int = 100):
        super().__init__(pipeline_space=pipeline_space, patience=patience)

    def sample(self, acquisition_function=None, constraint=None) -> SearchSpace:
        constraint = constraint or (lambda _: True)
        assert self.patience >= 0
        for _ in range(self.patience + 1):
            config = self.pipeline_space.sample(patience=self.patience)
            if constraint(config):
                return config
        return config
