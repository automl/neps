from __future__ import annotations

from typing import Iterable

from more_itertools import first

from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .base_acq_sampler import AcquisitionSampler
from .random_sampler import RandomSampler


class FreezeThawSampler(AcquisitionSampler):
    n = 1000  # number of random samples to draw at lowest fidelity

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observations = None

    def _sample_new(self, n: int = None):
        n = n if n is not None else self.n
        configs = [
            self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            for _ in range(n)
        ]
        for _config in configs:
            _config.fidelity.value = self.pipeline_space.lower  # assigns min budget
        return configs

    def sample(self, acquisition_function=None, n: int = None) -> list[SearchSpace]:
        # TODO: return dataframe with indices
        # collect partial curves
        # TODO: retrieve the observed configurations with their fidelities set to their max seen
        lcs = self.observations.get_partial_configs_at_max_seen()
        # TODO: this df should have the config IDs that they were assigned

        # collect multiple random samples
        # TODO: add a count of the number of samples
        # TODO: return dataframe with indices
        config = self._sample_new()
        # TODO: this df should have congig IDs that have not yet been assigned
        #    something like range(len(observed_configs), len(observed_configs) + n)

        # TODO: concatenate the two dataframes and return
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
        return
