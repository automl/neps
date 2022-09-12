from __future__ import annotations

import typing
from copy import deepcopy

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.hyperparameters.categorical import (
    CATEGORICAL_CONFIDENCE_SCORES,
    CategoricalParameter,
)
from ...search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES, FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import FixedPriorPolicy, RandomUniformPolicy


class SuccessiveHalving(BaseOptimizer):
    """Implements a SuccessiveHalving procedure with a sampling and promotion policy."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RandomUniformPolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
        )
        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        # SH implicitly sets early_stopping_rate to 0
        # the parameter is exposed to allow HB to call SH with different stopping rates
        self.early_stopping_rate = early_stopping_rate
        self.sampling_policy = sampling_policy(pipeline_space)
        self.promotion_policy = promotion_policy(self.eta)

        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type
        self.use_priors = use_priors

        # check to ensure no rung ID is negative
        # equivalent to s_max in https://arxiv.org/pdf/1603.06560.pdf
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit

        # maps rungs to a fidelity value for an SH bracket with `early_stopping_rate`
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        self.config_map = self._get_config_map(self.early_stopping_rate)

        self.min_rung = min(list(self.rung_map.keys()))
        self.max_rung = max(list(self.rung_map.keys()))

        # placeholder args for varying promotion and sampling policies
        self.promotion_policy_kwargs: dict = {}
        self.promotion_policy_kwargs.update({"config_map": self.config_map})
        self.sampling_args: dict = {}

        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))
        # stores which configs occupy each rung at any time
        self.rung_members: dict = {}  # stores config IDs per rung
        self.rung_members_performance: dict = dict()  # performances recorded per rung
        self.rung_promotions: dict = {}  # records a promotable config per rung
        self.total_fevals = 0

        # setup SH state counter
        self._counter = 0
        self.full_rung_trace = SuccessiveHalving._get_rung_trace(
            self.rung_map, self.config_map
        )

        # prior setups
        self.prior_confidence = prior_confidence
        self._enhance_priors()

    @classmethod
    def _get_rung_trace(cls, rung_map: dict, config_map: dict) -> list[int]:
        """Lists the rung IDs in sequence of the flattened SH tree."""
        rung_trace = []
        for rung in sorted(rung_map.keys()):
            rung_trace.extend([rung] * config_map[rung])
        return rung_trace

    def get_incumbent_score(self):
        y_star = np.inf  # minimizing optimizer
        if len(self.observed_configs):
            y_star = self.observed_configs.perf.values.min()
        return y_star

    def _update_state_counter(self) -> None:
        """Updates a counter to map where in the rung trace the current SH is."""
        self._counter += 1
        self._counter %= len(self.full_rung_trace)

    def _get_rung_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        rung = self.full_rung_trace[self._counter]
        return rung

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
            # TODO: add +s to keys and TEST
            rung_map[i + s] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, IntegerParameter)
                else _max_budget
            )
            _max_budget /= self.eta
        return rung_map

    def _get_config_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to the number of configs for each fidelity"""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        s_max = self.stopping_rate_limit + 1
        _s = self.stopping_rate_limit - s
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        config_map = dict()
        for i in range(nrungs):
            # TODO: add +s to keys and TEST
            config_map[i + s] = int(_n_config)
            _n_config //= self.eta
        return config_map

    @classmethod
    def _get_config_id_split(cls, config_id: str) -> tuple[str, str]:
        # assumes config IDs of the format `[unique config int ID]_[int rung ID]`
        _config, _rung = config_id.split("_")
        return _config, _rung

    # TODO: check pending
    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = self.get_loss(
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

    def _handle_pending_evaluations(
        self, pending_evaluations: dict[str, ConfigResult]
    ) -> None:
        # iterates over all pending evaluations and updates the list of observed
        # configs with the rung and performance as None
        for config_id, _ in pending_evaluations.items():
            _config, _rung = self._get_config_id_split(config_id)
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
                self.observed_configs.at[int(_config), "perf"] = None
        return

    def _get_rungs_state(self):
        # to account for incomplete evaluations from being promoted --> working on a copy
        _observed_configs = self.observed_configs.copy().dropna(inplace=False)
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in self.rung_map.keys()}
        self.rung_members_performance = {k: [] for k in self.rung_map.keys()}
        for _rung in _observed_configs.rung.unique():
            self.rung_members[_rung] = _observed_configs.index[
                _observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = _observed_configs.perf[
                _observed_configs.rung == _rung
            ].values
        return

    def _handle_promotions(self):
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        self.rung_promotions = self.promotion_policy.retrieve_promotions()

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        """This is basically the fit method.

        Args:
            previous_results (dict[str, ConfigResult]): [description]
            pending_evaluations (dict[str, ConfigResult]): [description]
        """
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(previous_results)
        self.total_fevals = len(previous_results)

        # account for pending evaluations
        self._handle_pending_evaluations(pending_evaluations)

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # identifying promotion list per rung
        self._handle_promotions()
        return

    def is_promotable(self) -> int | None:
        """Returns an int if a rung can be promoted, else a None."""
        rung_next = self._get_rung_to_run()
        rung_to_promote = rung_next - 1
        if (
            rung_to_promote >= self.min_rung
            and len(self.rung_promotions[rung_to_promote]) > 0
        ):
            return rung_to_promote
        return None

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search.

        If `initial_design_type` is `max_budget`, the number of evaluations made at the
            target budget is compared to `initial_design_size`.
        If `initial_design_type` is `unique_configs`, the total number of unique
            configurations is compared to the `initial_design_size`.
        """
        _observed_configs = self.observed_configs.copy().dropna()
        if self.initial_design_type == "max_budget":
            val = (
                np.sum(_observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            val = len(_observed_configs) <= self._initial_design_size
        return val

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promotes the first recorded promotable config in the argsort-ed rung
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[rung]
            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
        else:
            if self.sampling_policy is None:
                config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=self.use_priors,
                    ignore_fidelity=True,
                )
            else:
                config = self.sampling_policy.sample(**self.sampling_args)
            # TODO: make base rung depend on rung_map
            rung_id = self.min_rung
            fidelity_value = self.rung_map[rung_id]
            config.fidelity.value = fidelity_value

            previous_config_id = None
            config_id = f"{len(self.observed_configs)}_{rung_id}"

        # IMPORTANT to tell synchronous SH to query the next allocation
        self._update_state_counter()
        return config.hp_values(), config_id, previous_config_id  # type: ignore

    def _enhance_priors(self):
        """Only applicable when priors are given along with a confidence."""
        if not self.use_priors and self.prior_confidence is None:
            return
        for k in self.pipeline_space.keys():
            if self.pipeline_space[k].is_fidelity:
                continue
            elif isinstance(self.pipeline_space[k], (FloatParameter, IntegerParameter)):
                confidence = FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(self.pipeline_space[k], CategoricalParameter):
                confidence = CATEGORICAL_CONFIDENCE_SCORES[self.prior_confidence]
                self.pipeline_space[k].default_confidence_score = confidence


class SuccessiveHalvingWithPriors(SuccessiveHalving):
    """Implements a SuccessiveHalving procedure with a sampling and promotion policy."""

    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
        )


class AsynchronousSuccessiveHalving(SuccessiveHalving):
    """Implements ASHA with a sampling and asynchronous promotion policy."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RandomUniformPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from SH
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            use_priors=use_priors,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
        )

    def is_promotable(self) -> int | None:
        """Returns an int if a rung can be promoted, else a None."""
        rung_to_promote = None
        # iterates starting from the highest fidelity promotable to the lowest fidelity
        for rung in reversed(range(self.min_rung, self.max_rung)):
            if len(self.rung_promotions[rung]) > 0:
                rung_to_promote = rung
                # stop checking when a promotable config found
                # no need to search at lower fidelities
                break
        return rung_to_promote


class AsynchronousSuccessiveHalvingWithPriors(AsynchronousSuccessiveHalving):
    """Implements ASHA with a sampling and asynchronous promotion policy."""

    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from SH
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
        )


if __name__ == "__main__":
    pass
