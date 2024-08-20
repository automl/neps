from __future__ import annotations

import torch
from neps.search_spaces import SearchSpace
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)


class RandomSampler(AcquisitionSampler):

    def sample(self, n: int, space: SearchSpace) -> torch.Tensor:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=False
        )
