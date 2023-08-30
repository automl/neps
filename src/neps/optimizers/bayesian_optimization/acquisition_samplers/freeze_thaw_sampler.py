# type: ignore
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .base_acq_sampler import AcquisitionSampler


class FreezeThawSampler(AcquisitionSampler):
    n = 1000  # number of random samples to draw at lowest fidelity

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
        return pd.Series(configs, index=range(index_from, index_from + len(configs)))

    def sample(self, acquisition_function=None, n: int = None) -> pd.Series:
        partial_configs = self.observations.get_partial_configs_at_max_seen()

        new_configs = self._sample_new(index_from=self.observations.next_config_id(), n=n)

        # Set fidelity for observed configs
        partial_configs_list = []
        index_list = []
        for idx, config in partial_configs.items():
            next_fidelity = config.fidelity.value + self.b_step
            # Select only the configs not exceeding the max budget
            if np.less_equal(next_fidelity, config.fidelity.upper):
                _config = deepcopy(config)
                _config.fidelity.value = next_fidelity
                partial_configs_list.append(_config)
                index_list.append(idx)

        # We build a new series of partial configs to avoid
        # incrementing fidelities multiple times due to pass-by-reference
        partial_configs = pd.Series(partial_configs_list, index=index_list)

        # Set fidelity for new configs
        for _, config in new_configs.items():
            config.fidelity.value = config.fidelity.lower

        configs = pd.concat([partial_configs, new_configs])

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
