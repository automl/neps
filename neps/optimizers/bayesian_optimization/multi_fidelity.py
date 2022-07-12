from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Any, Callable

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult, instance_from_map
from typing_extensions import Literal

from ...search_spaces.numerical.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ...utils.result_utils import get_cost, get_loss
from .optimizer import BayesianOptimization


class BaseMultiFidelityOptimization(BayesianOptimization):
    """Base optimizer for multi-fidelity optimization"""

    USES_COST_MODEL = False
    USES_CONTINUATION = True
    # TODO: add param to allow non-discrete fidelity space

    @dataclass
    class ObservedRecord:
        base_id: int
        fidelity_step: int
        config: SearchSpace
        loss: float  # loss of the configuration with the max fidelity
        cost: float  # cost of the configuration with the max fidelity
        trace: dict
        min_loss: float
        step_min_loss: int
        config_min_loss: SearchSpace

    def __init__(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: str | Any = "gp",
        num_fidelity_steps: int = 10,
        aggregate_continuation_costs: Callable | Literal["sum", "max"] | None = None,
        **kwargs,
    ):
        if num_fidelity_steps < 1 or not isinstance(num_fidelity_steps, int):
            raise ValueError(
                "num_fidelity_steps should be at an integer with a value of least 1"
            )

        if aggregate_continuation_costs is None:
            # Choose a good default value if not given
            aggregate_continuation_costs = "sum" if self.USES_CONTINUATION else "max"

        # Set the choices for the pipeline_space fidelity values
        fid_choices = [
            pipeline_space.fidelity.from_step(step, num_fidelity_steps, in_place=False)
            for step in range(num_fidelity_steps)
        ]
        if pipeline_space.fidelity.choices is None:
            pipeline_space.fidelity.choices = fid_choices
        elif pipeline_space.fidelity.choices != fid_choices:
            raise Exception("Can't use a fidelity with choices different from the rungs")

        super().__init__(
            pipeline_space=pipeline_space,
            surrogate_model=surrogate_model,
            **kwargs,
        )

        self.observed_configs: pd.DataFrame = None
        self.fantasized_remaining_budget = self.budget
        self.num_fidelity_steps: int = num_fidelity_steps
        self.aggregate_continuation_costs: Callable = instance_from_map(
            {"sum": lambda a, b: a + b, "max": max},
            aggregate_continuation_costs,
            "aggregate_continuation_costs",
        )

        # maintain incumbent across 3 dimensions:
        #  cost - the configuration with the maximum cumulative cost incurred
        #  fidelity_step - the configuration with the highest fidelity seen
        #  loss - the configuration scoring the lowest loss
        self.incumbent = dict(cost=None, fidelity_step=None, score=None)

    def _update_incumbents(self):
        if self.observed_configs is None or len(self.observed_configs) == 0:
            return
        # stores the IDs of the entry in the data structure `observed_configs`
        self.incumbent.update(
            cost=self.observed_configs.index.values[
                np.argmax(self.observed_configs.cost.values)
            ],
            fidelity_step=self.observed_configs.index.values[
                np.argmax(self.observed_configs.fidelity_step.values)
            ],
            loss=self.observed_configs.index.values[
                np.argmin(self.observed_configs.loss.values)
            ],
        )

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        """Load the results, and ensure the results cost are the total cost and not
        only the continuation cost if continuation is used."""
        results_by_base_id = defaultdict(lambda: [])
        records: list[BaseMultiFidelityOptimization.ObservedRecord] = []

        for config_id, result in previous_results.items():
            base_cfg_id, fidelity_step = map(int, config_id.split("_"))
            if result.result != "error":
                results_by_base_id[base_cfg_id].append((fidelity_step, result))

        for base_cfg_id, all_config_vals in results_by_base_id.items():
            all_config_vals.sort(key=lambda fid_cfg: fid_cfg[0])
            last_fidelity_step, last_config = all_config_vals[-1]

            costs_list = [get_cost(config.result) for _, config in all_config_vals]
            costs_list = list(accumulate(costs_list, self.aggregate_continuation_costs))

            # Setting the real aggregated cost for configuratioons
            for (_, config), real_cost in zip(all_config_vals, costs_list):
                config.result["cost"] = real_cost

            cost = costs_list[-1]
            trace = {
                fidelity_step: config_val for fidelity_step, config_val in all_config_vals
            }

            all_config_vals.sort(key=lambda fid_cfg: get_loss(fid_cfg[1].result))
            min_loss = get_loss(all_config_vals[-1][1].result)
            step_min_loss = all_config_vals[-1][0]
            config_min_loss = all_config_vals[-1][1].config

            records.append(
                self.ObservedRecord(
                    base_id=base_cfg_id,
                    fidelity_step=last_fidelity_step,
                    config=last_config.config,
                    min_loss=min_loss,
                    step_min_loss=step_min_loss,
                    config_min_loss=config_min_loss,
                    loss=get_loss(last_config.result),
                    cost=cost,
                    trace=trace,
                )
            )
        if records:
            self.observed_configs = pd.DataFrame(records)
            self.observed_configs.set_index("base_id", inplace=True)
            self.observed_configs.sort_index(inplace=True)

        super().load_results(previous_results, pending_evaluations)

    def _update_optimizer_training_state(self) -> None:
        """Called inside load_results()."""
        super()._update_optimizer_training_state()
        self._update_incumbents()

        # TODO: subtract optimizer overhead too? -> parameterize this behaviour
        self.fantasized_remaining_budget = self.remaining_budget
        if self.USES_COST_MODEL:
            assert self.fantasized_costs is not None
            self.fantasized_remaining_budget -= sum(self.fantasized_costs)

    def get_new_config_id(self, config, base_id=None, fidelity_step=None):
        """An id should be of the form [base_id]_[fidelity_step], with the same
        base_id being shared by configuration with the same parameter values,
        except for the fidelity value.
        """
        if base_id is None:
            base_id = 1 + max(
                (
                    int(prev_id.split("_")[0])
                    for prev_id in chain(
                        self._previous_results.keys(), self._pending_evaluations.keys()
                    )
                ),
                default=0,
            )
        if fidelity_step is None:
            fidelity_step = config.fidelity.step_on_scale(self.num_fidelity_steps)
        return f"{base_id}_{fidelity_step}"

    def sample_configuration_randomly(self, *args, **kwargs):
        # We want the configurations for the initialization to have the lowest fidelity
        config, config_id, previous_id = super().sample_configuration_randomly(
            *args, **kwargs
        )
        config.fidelity.value = config.fidelity.lower
        return config, config_id, previous_id


class BayesianOptimizationMultiFidelity(BayesianOptimization):
    # TODO: inherit BaseMultiFidelityOptimization? Clean the code
    def __init__(
        self,
        pipeline_space: SearchSpace,
        eta: int = 4,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        model_search: bool = True,
        switch_to_bo: bool = False,
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
        self.model_search = model_search
        self.switch_to_bo = switch_to_bo

        # check to ensure no rung ID is negative
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit

        # maps rungs to a fidelity value
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        print(self.rung_map)
        self.max_rung = len(self.rung_map) - 1
        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "loss"))
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

    # TODO: check pending
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
                    self.observed_configs.at[int(_config), "loss"] = self.get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), self.get_loss(config_val.result)]],
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
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "loss"))

        if len(previous_results) > 0 and len(self.observed_configs) == 0:
            # previous optimization run exists and needs to be loaded
            self._load_previous_observations(previous_results)

        if not self.is_init_phase():
            # to build surrogate only after initial design
            super().load_results(previous_results, pending_evaluations)

        self.total_fevals = len(previous_results)
        # iterates over all previous results and updates the list of observed
        # configs with the highest fidelity it was evaluated on and its performance
        for config_id, config_val in previous_results.items():
            _config, _rung = config_id.split("_")
            loss = self.get_loss(config_val.result)
            if int(_config) not in self.observed_configs.index:
                # this condition and check is important to handle async scenarios as
                # the `previous_results` can provide configs that have not been
                # encountered by this instantiation of the optimizer object
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), loss]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                if int(_rung) >= self.observed_configs.at[int(_config), "rung"]:
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "loss"] = loss
        # iterates over all pending evaluations and updates the list of observed
        # configs with the rung and performance as None
        for config_id, _ in pending_evaluations.items():
            _config, _rung = config_id.split("_")
            if int(_config) not in self.observed_configs.index:
                _df = pd.DataFrame(
                    [[None, int(_rung), None]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                self.observed_configs.at[int(_config), "rung"] = int(_rung)
                self.observed_configs.at[int(_config), "loss"] = None

        # to account for incomplete evaluations from being promoted
        _observed_configs = self.observed_configs.copy().dropna(inplace=False)
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in range(self.max_rung)}
        self.rung_members_performance = {k: [] for k in range(self.max_rung)}
        for _rung in _observed_configs.rung.unique():
            self.rung_members[_rung] = _observed_configs.index[
                _observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = _observed_configs.loss[
                _observed_configs.rung == _rung
            ].values
        # identifying promotion list per rung
        self.rung_promotions = dict()
        for _rung in self.rung_map.keys():
            if _rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            top_k = len(self.rung_members_performance[_rung]) // self.eta
            self.rung_promotions[_rung] = []
            if top_k > 0:
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
        _observed_configs = self.observed_configs.copy().dropna()
        if self.initial_design_type == "max_budget":
            return (
                np.sum(_observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            return super().is_init_phase()

    def _switch_to_bo(self):
        """Switches to BO when eta evaluations seen at the highest fidelity"""
        max_fidelity_evals = np.sum(self.observed_configs.rung == self.max_rung)
        if self.switch_to_bo and max_fidelity_evals >= self.eta:
            print("\nSwitching to BO!\n")
            fidelity_value = self.rung_map[self.max_rung]  # base rung is always 0
            config_id = f"{len(self.observed_configs)}_{self.max_rung}"
        else:
            fidelity_value = self.rung_map[0]  # base rung is always 0
            config_id = f"{len(self.observed_configs)}_{0}"
        return fidelity_value, config_id

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
        else:
            if random.random() < self._random_interleave_prob:
                config = self.random_sampler.sample(
                    constraint=self.sampling_constraint, user_priors=False
                )
            elif self.model_search and not self.is_init_phase():
                # sampling from AF at base rung
                for _ in range(self.patience):
                    config = self.acquisition_sampler.sample(
                        self.acquisition, constraint=self.sampling_constraint
                    )
                    if config not in self._pending_evaluations.values():
                        break
                else:
                    # random sampling a config if failed to sample a new config from AF
                    config = self.random_sampler.sample(
                        user_priors=True, constraint=self.sampling_constraint
                    )
            else:
                # random sampling a config
                config = self.random_sampler.sample(
                    user_priors=True, constraint=self.sampling_constraint
                )
            # assigning the fidelity to evaluate the config at
            config.fidelity.value, config_id = self._switch_to_bo()
            previous_config_id = None
        return config, config_id, previous_config_id  # type: ignore
