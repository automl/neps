from __future__ import annotations

from typing import Callable
import warnings

import numpy as np
import pandas as pd
from copy import deepcopy

from neps.search_spaces.search_space import SearchSpace
from neps.optimizers.multi_fidelity.utils import MFObservedData
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)


SAMPLES_TO_DRAW = (
    100  # number of random samples to draw for optimizing acquisition function
)


class FreezeThawSampler(AcquisitionSampler):
    def __init__(self, samples_to_draw: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.observations = None
        self.b_step = None
        self.n = None
        self.pipeline_space = None
        # args to manage tabular spaces/grid
        self.is_tabular = False  # flag is set by `set_state()`
        self.sample_full_table = None
        self.samples_to_draw = (
            samples_to_draw if samples_to_draw is not None else SAMPLES_TO_DRAW
        )
        self.set_sample_full_tabular(True)  # sets flag that samples full table

    def set_sample_full_tabular(self, flag: bool = False):
        if self.is_tabular:
            self.sample_full_table = flag

    def _sample_new(
        self,
        index_from: int,
        n: int | None = None,
        ignore_fidelity: bool = False,
    ) -> pd.Series:
        n = n if n is not None else self.samples_to_draw
        assert self.pipeline_space is not None
        new_configs = [
            self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=ignore_fidelity
            )
            for _ in range(n)
        ]

        return pd.Series(
            new_configs, index=range(index_from, index_from + len(new_configs))
        )

    def _sample_new_unique(
        self,
        index_from: int,
        n: int | None = None,
        patience: int = 10,
        ignore_fidelity: bool = False,
    ) -> pd.Series:
        n = n if n is not None else self.samples_to_draw
        assert (
            patience > 0 and n > 0
        ), "Patience and `samples_to_draw` must be larger than 0"

        assert self.observations is not None
        assert self.pipeline_space is not None

        existing_configs = self.observations.all_configs_list()
        new_configs = []
        for _ in range(n):
            # Sample patience times for an unobserved configuration
            for _ in range(patience):
                _config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=False,
                    ignore_fidelity=ignore_fidelity,
                )
                # # Convert continuous into tabular if the space is tabular
                # _config = continuous_to_tabular(_config, self.tabular_space)
                # Iterate over all observed configs
                for config in existing_configs:
                    if _config.is_equal_value(
                        config, include_fidelity=not ignore_fidelity
                    ):
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

    def sample(
        self,
        acquisition_function: Callable | None = None,
        n: int | None = None,
        set_new_sample_fidelity: int | float | None = None,
    ) -> pd.DataFrame:
        """Samples a new set and returns the total set of observed + new configs."""
        assert self.observations is not None
        assert self.pipeline_space is not None
        assert self.pipeline_space.custom_grid_table is not None

        partial_configs = self.observations.get_partial_configs_at_max_seen()

        _n = n if n is not None else self.samples_to_draw
        if self.is_tabular:
            # handles tabular data such that the entire unseen set of configs from the
            # table is considered to be the new set of candidates
            _partial_ids = {conf["id"].value for conf in partial_configs}
            _all_ids = set(list(self.pipeline_space.custom_grid_table.keys()))

            # accounting for unseen configs only, samples remaining table if flag is set
            max_n = len(_all_ids) + 1 if self.sample_full_table else _n
            _n = min(max_n, len(_all_ids - _partial_ids))

            _new_configs = np.random.choice(
                list(_all_ids - _partial_ids), size=_n, replace=False
            )
            placeholder_config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            _configs = [placeholder_config.clone() for _id in _new_configs]
            for _i, val in enumerate(_new_configs):
                _configs[_i]["id"].set_value(val)

            new_configs = pd.Series(
                _configs,
                index=np.arange(
                    len(partial_configs), len(partial_configs) + len(_new_configs)
                ),
            )
        else:
            # handles sampling new configurations for continuous spaces
            new_configs = self._sample_new(
                index_from=self.observations.next_config_id(), n=_n, ignore_fidelity=False
            )
            # Continuous benchmarks need to deepcopy individual configs here,
            # because in contrast to tabular benchmarks
            # they are not reset in every sampling step

            # TODO: I do not know what the f p_config_ is meant to be so I don't know
            # if we have a specific clone method or not...
            partial_configs = pd.Series(
                [deepcopy(p_config_) for idx, p_config_ in partial_configs.items()],
                index=partial_configs.index,
            )

        # Updating fidelity values
        if set_new_sample_fidelity is not None:
            for config in new_configs:
                config.update_hp_values({config.fidelity_name: set_new_sample_fidelity})

        configs = pd.concat([deepcopy(partial_configs), new_configs])

        return configs  # type: ignore

    def set_state(
        self,
        pipeline_space: SearchSpace,
        observations: MFObservedData,
        b_step: int,
        n: int | None = None,
    ) -> None:
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.observations = observations
        self.b_step = b_step
        self.n = n if n is not None else self.samples_to_draw
        if (
            hasattr(self.pipeline_space, "custom_grid_table")
            and self.pipeline_space.custom_grid_table is not None
        ):
            self.is_tabular = True
            self.set_sample_full_tabular(True)
