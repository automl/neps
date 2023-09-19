# type: ignore
from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData, continuous_to_tabular
from .base_acq_sampler import AcquisitionSampler


class FreezeThawSampler(AcquisitionSampler):
    SAMPLES_TO_DRAW = 100  # number of random samples to draw at lowest fidelity

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observations = None
        self.b_step = None
        self.n = None
        self.tabular_space = None

    def _sample_new(
        self, index_from: int, n: int = None, patience: int = 10
    ) -> pd.Series:
        n = n if n is not None else self.SAMPLES_TO_DRAW
        assert (
            patience > 0 and n > 0
        ), "Patience and SAMPLES_TO_DRAW must be larger than 0"

        existing_configs = self.observations.all_configs_list()
        new_configs = []
        for _ in range(n):
            # Sample patience times for an unobserved configuration
            for _ in range(patience):
                _config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=False, ignore_fidelity=False
                )
                # Convert continuous into tabular if the space is tabular
                _config = continuous_to_tabular(_config, self.tabular_space)
                # Iterate over all observed configs
                for config in existing_configs:
                    if _config.is_equal_value(config, include_fidelity=False):
                        # if the sampled config already exists
                        # do the next iteration of patience
                        break
                else:
                    # If the new sample is not equal to any previous
                    # then it's a new config
                    new_config = _config
                    break
            else:
                # TODO: use logger.warn here instead (karibbov)
                warnings.warn(
                    f"Couldn't find an unobserved configuration in {patience} "
                    f"iterations. Using an observed config instead"
                )
                # patience budget exhausted use the last sampled config anyway
                new_config = _config

            # append the new config to the list
            new_configs.append(new_config)

        return pd.Series(
            new_configs, index=range(index_from, index_from + len(new_configs))
        )

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
        tabular_space: SearchSpace,
        n: int = None,
    ):
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.observations = observations
        self.b_step = b_step
        self.n = n if n is not None else self.SAMPLES_TO_DRAW
        self.tabular_space = tabular_space
