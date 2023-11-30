from __future__ import annotations

from ....search_spaces.search_space import SearchSpace
from .base_acq_sampler import AcquisitionSampler


class RandomSampler(AcquisitionSampler):
    def __init__(self, pipeline_space: SearchSpace, patience: int = 100):
        super().__init__(pipeline_space=pipeline_space, patience=patience)

    def sample(self, acquisition_function=None) -> list[SearchSpace]:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=False
        )
