from __future__ import annotations

from ..base_optimizer import BaseOptimizer
from ..bayesian_optimization.acquisition_samplers.random_sampler import RandomSampler


class RandomSearch(BaseOptimizer):
    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self.random_sampler = RandomSampler()

    def sample(self):
        return self.random_sampler.sample()

    def _update_model(self):
        self.random_sampler.work_with(self.pipeline_space)
