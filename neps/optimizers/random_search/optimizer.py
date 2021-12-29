from __future__ import annotations

import random

from deprecated import deprecated

from ..base_optimizer import Optimizer
from ..bayesian_optimization.acquisition_function_optimization.random_sampler import (
    RandomSampler,
)


class RandomSearch(Optimizer):
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
        return config, f"{self._len_previous}_{self._len_pending}", None

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        self._len_previous = len(previous_results)
        self._len_pending = len(pending_evaluations)

    @deprecated
    def _initialize_model(self, **kwargs):
        pass

    @deprecated
    def _update_model(self, **kwargs):
        pass

    @deprecated
    def _propose_new_location(
        self, batch_size: int = 5, n_candidates: int = 10
    ) -> tuple[list, dict | None]:
        # create candidate pool
        pool = self.random_sampler.sample(n_candidates)

        next_x = random.sample(pool, batch_size)
        self.sampled_idx.append(next_x)

        opt_details = {"pool": pool} if self.return_opt_details else None

        return next_x, opt_details

    @deprecated
    def get_config(self):
        config = self.random_sampler.sample(1)[0]
        return config

    @deprecated
    def new_result(self, job):
        # if job.result is None:
        #     loss = np.inf
        # else:
        #     loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        #
        # config = job.kwargs["config"]
        pass
