from __future__ import annotations

import metahyper

from ..bayesian_optimization.acquisition_function_optimization.random_sampler import (
    RandomSampler,
)


class RandomSearch(metahyper.Sampler):
    def __init__(self, pipeline_space, return_opt_details: bool = False):
        super().__init__()
        self.sampled_idx: list = []
        self.return_opt_details = return_opt_details
        self.surrogate_model = None
        self.random_sampler = RandomSampler(pipeline_space)
        self._len_previous = 0
        self._len_pending = 0

    def get_config_and_ids(self):
        config = self.random_sampler.sample(1)[0]
        config_id = str(self._len_previous + self._len_pending + 1)
        return config, config_id, None

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        self._len_previous = len(previous_results)
        self._len_pending = len(pending_evaluations)
