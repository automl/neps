from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.numerical.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ...utils.result_utils import get_loss
from .optimizer import BayesianOptimization


class BayesianOptimizationMultiFidelity(BayesianOptimization):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        eta: int = 4,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "unique_configs",
        **bo_kwargs,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            **bo_kwargs,
        )
        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        self.early_stopping_rate = early_stopping_rate
        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type

        # check to ensure no rung ID is negative
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit

        # maps rungs to a fidelity value
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        self.max_rung = len(self.rung_map) - 1
        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))
        # stores which configs occupy each rung at any time
        self.rung_members: dict = {k: [] for k in self.rung_map.keys()}
        # one-to-one with self.rung_members, storing corresponding performance
        self.rung_members_performance: dict = {k: [] for k in self.rung_map.keys()}
        self.rung_promotions: dict = dict()
        self.total_fevals = 0

    def _get_rung_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to a fidelity value based on fidelity bounds, eta, s."""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        _max_budget = self.max_budget
        rung_map = dict()
        for i in reversed(range(nrungs)):
            rung_map[i] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, IntegerParameter)
                else _max_budget
            )
            _max_budget /= self.eta
        return rung_map

    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = config_id.split("_")
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), get_loss(config_val.result)]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        return

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        super().load_results(previous_results, pending_evaluations)

        if len(previous_results) > 0 and len(self.observed_configs) == 0:
            # previous optimization run exists and needs to be loaded
            self._load_previous_observations(previous_results)

        self.total_fevals = len(previous_results)
        # iterates over all previous results and updates the list of observed
        # configs with the highest fidelity it was evaluated on and its performance
        for config_id, _ in previous_results.items():
            # any config occurring in `previous_results` should be in `observed_configs`
            _config, _rung = config_id.split("_")
            # `max` is important to keep track of the performance of a configuration on
            # the highest fidelity seen as `previous_results` contain all evaluations
            _rung = max(int(_rung), self.observed_configs.iloc[int(_config)]["rung"])
            # _config = int(_config)
            self.observed_configs.at[int(_config), "rung"] = _rung
            perf = get_loss(previous_results[f"{int(_config)}_{_rung}"].result)
            self.observed_configs.at[int(_config), "perf"] = perf
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in range(self.max_rung)}
        self.rung_members_performance = {k: [] for k in range(self.max_rung)}
        for _rung in self.observed_configs.rung.unique():
            self.rung_members[_rung] = self.observed_configs.index[
                self.observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = self.observed_configs.perf[
                self.observed_configs.rung == _rung
            ].values
        # identifying promotion list per rung
        self.rung_promotions = dict()
        for _rung in self.rung_map.keys():
            if _rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            top_k = len(self.rung_members_performance[_rung]) // self.eta
            self.rung_promotions[_rung] = np.array(self.rung_members[_rung])[
                np.argsort(self.rung_members_performance[_rung])[:top_k]
            ].tolist()
        return

    def is_promotable(self) -> int | None:
        rung_to_promote = None
        for _rung in reversed(range(self.max_rung)):
            if len(self.rung_promotions[_rung]) > 0:
                rung_to_promote = _rung
                break
        return rung_to_promote

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search.

        If `initial_design_type` is `max_budget`, the number of evaluations made at the
            target budget is compared to `initial_design_size`.
        If `initial_design_type` is `unique_configs`, the total number of unique
            configurations is compared to the `initial_design_size`.
        """
        if self.initial_design_type == "max_budget":
            val = (
                np.sum(self.observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            val = len(self.observed_configs) <= self._initial_design_size
        return val

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promote existing config
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[rung]
            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
            # updating observation tracker
            self.observed_configs.at[row.name, "rung"] = rung
        else:
            if self.is_init_phase():
                # random sampling a config at base rung
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )
            else:
                # sampling from AF at base rung
                config = self.acquisition_sampler.sample(self.acquisition)
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[0]  # base rung is always 0
            # updating observation tracker
            _df = pd.DataFrame(
                [[config, 0, None]],
                columns=self.observed_configs.columns,
                index=pd.Series(len(self.observed_configs)),  # key for config_id
            )
            self.observed_configs = pd.concat((self.observed_configs, _df))
            # updating config IDs
            config_id = f"{len(self.observed_configs) - 1}_{0}"
            previous_config_id = None
        return config, config_id, previous_config_id  # type: ignore
