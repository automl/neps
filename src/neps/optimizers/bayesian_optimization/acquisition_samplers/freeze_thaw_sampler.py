# type: ignore
from __future__ import annotations

import pandas as pd

from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .base_acq_sampler import AcquisitionSampler


class FreezeThawSampler(AcquisitionSampler):
    n = 500  # number of random samples to draw at lowest fidelity

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observations = None
        self.b_step = None

    def _sample_new(self, index_from: int, n: int = None) -> pd.Series:
        n = n if n is not None else self.n
        configs = [
            self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            for _ in range(n)
        ]
        for _config in configs:
            _config.fidelity.value = (
                self.pipeline_space.fidelity.lower
            )  # assigns min budget

        return pd.Series(configs, index=range(index_from, index_from + len(configs)))

    def sample(self, acquisition_function=None, n: int = None) -> pd.Series:
        lcs = self.observations.get_partial_configs_at_max_seen()

        configs = self._sample_new(index_from=self.observations.next_config_id(), n=n)
        configs = pd.concat([lcs, configs])

        return configs

    def set_state(
        self,
        pipeline_space: SearchSpace,
        observations: MFObservedData,
        b_step: int,
        n: int = None,
    ):
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.observations = observations
        self.b_step = b_step
        self.n = n if n is not None else self.n
