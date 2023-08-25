from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from more_itertools import first

from .base_acq_sampler import AcquisitionSampler
from .random_sampler import RandomSampler


class FreezeThawSampler(AcquisitionSampler):
    n = 1000

    def __init__(
        self,
    ):
        self.observations = None

    def _sample_new(self):
        configs = [
            self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            for _ in range(self.n)
        ]
        for _config in configs:
            _config.fidelity.value = self.pipeline_space.lower
        return configs

    def sample(self, acquisition_function=None) -> list[SearchSpace]:
        # TODO: return dataframe with indices
        # collect partial curves
        # TODO: retrieve the partial curves correctly
        lcs = self.observations.get_partial_configs_at_max_seen()

        # collect multiple random samples
        # TODO: add a count of the number of samples
        config = self._sample_new()

        # collate the two sets
        configs = lcs + configs

        return configs

    def set_state(
        self,
        pipeline_space: SearchSpace,
        observations: MFObservedData,
        b_step: int,
        n: int = None,
        **kwargs,
    ):
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.observations = observations
        self.b_step = b_step
        self.n = n if n is not None else self.n
        return super().set_state(surrogate_model, **kwargs)
