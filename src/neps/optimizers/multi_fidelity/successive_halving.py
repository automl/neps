# type: ignore

from __future__ import annotations

import random
import typing
from copy import deepcopy

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper import ConfigResult

from ...search_spaces.hyperparameters.categorical import (
    CATEGORICAL_CONFIDENCE_SCORES,
    CategoricalParameter,
)
from ...search_spaces.hyperparameters.constant import ConstantParameter
from ...search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES, FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import FixedPriorPolicy, RandomUniformPolicy

CUSTOM_FLOAT_CONFIDENCE_SCORES = FLOAT_CONFIDENCE_SCORES.copy()
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = CATEGORICAL_CONFIDENCE_SCORES.copy()
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})


class SuccessiveHalvingBase(BaseOptimizer):
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
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
    ):
        """Initialise an SH bracket.

        Args:
            pipeline_space: Space in which to search
            budget: Maximum budget
            eta: The reduction factor used by SH
            early_stopping_rate: Determines the number of rungs in an SH bracket
                Choosing 0 creates maximal rungs given the fidelity bounds
            initial_design_type: Type of initial design to switch to BO
                Legacy parameter from NePS BO design. Could be used to extend to MF-BO.
            use_priors: Allows random samples to be generated from a default
                Samples generated from a Gaussian centered around the default value
            sampling_policy: The type of sampling procedure to use
            promotion_policy: The type of promotion procedure to use
            loss_value_on_error: Setting this and cost_value_on_error to any float will
                supress any error during bayesian optimization and will use given loss
                value instead. default: None
            cost_value_on_error: Setting this and loss_value_on_error to any float will
                supress any error during bayesian optimization and will use given cost
                value instead. default: None
            logger: logger object, or None to use the neps logger
            prior_confidence: The range of confidence to have on the prior
                The higher the confidence, the smaller is the standard deviation of the
                prior distribution centered around the default
            random_interleave_prob: Chooses the fraction of samples from random vs prior
            sample_default_first: Whether to sample the default configuration first
            sample_default_at_target: Whether to evaluate the default configuration at
                the target fidelity or max budget
        """
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
        )
        if random_interleave_prob < 0 or random_interleave_prob > 1:
            raise ValueError("random_interleave_prob should be in [0.0, 1.0]")
        self.random_interleave_prob = random_interleave_prob
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        self.min_budget = self.pipeline_space.fidelity.lower
        self.max_budget = self.pipeline_space.fidelity.upper
        self.eta = eta
        # SH implicitly sets early_stopping_rate to 0
        # the parameter is exposed to allow HB to call SH with different stopping rates
        self.early_stopping_rate = early_stopping_rate
        self.sampling_policy = sampling_policy(
            pipeline_space=self.pipeline_space,
            logger=self.logger
        )
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
        self.rung_members: dict = dict()  # stores config IDs per rung
        self.rung_members_performance: dict = dict()  # performances recorded per rung
        self.rung_promotions: dict = dict()  # records a promotable config per rung
        self.total_fevals = 0

        # setup SH state counter
        self._counter = 0
        self.full_rung_trace = SuccessiveHalving._get_rung_trace(
            self.rung_map, self.config_map
        )

        #############################
        # Setting prior confidences #
        #############################
        # the std. dev or peakiness of distribution
        self.prior_confidence = prior_confidence
        self._enhance_priors()
        self.rung_histories = None

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
            config_map[i + s] = int(_n_config)
            _n_config //= self.eta
        return config_map

    @classmethod
    def _get_config_id_split(cls, config_id: str) -> tuple[str, str]:
        # assumes config IDs of the format `[unique config int ID]_[int rung ID]`
        _config, _rung = config_id.split("_")
        return _config, _rung

    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            perf = self.get_loss(config_val.result)
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "config"] = config_val.config
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = perf
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), perf]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            # for efficiency, redefining the function to have the
            # `rung_histories` assignment inside the for loop
            # rung histories are collected only for `previous` and not `pending` configs
            self.rung_histories[int(_rung)]["config"].append(int(_config))
            self.rung_histories[int(_rung)]["perf"].append(perf)
        return

    def _handle_pending_evaluations(
        self, pending_evaluations: dict[str, ConfigResult]
    ) -> None:
        # iterates over all pending evaluations and updates the list of observed
        # configs with the rung and performance as None
        for config_id, config in pending_evaluations.items():
            _config, _rung = self._get_config_id_split(config_id)
            if int(_config) not in self.observed_configs.index:
                _df = pd.DataFrame(
                    [[config, int(_rung), np.nan]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                self.observed_configs.at[int(_config), "rung"] = int(_rung)
                self.observed_configs.at[int(_config), "perf"] = np.nan
        return

    def clean_rung_information(self):
        self.rung_members = {k: [] for k in self.rung_map.keys()}
        self.rung_members_performance = {k: [] for k in self.rung_map.keys()}
        self.rung_promotions = {k: [] for k in self.rung_map.keys()}

    def _get_rungs_state(self, observed_configs=None):
        """Collects info on configs at a rung and their performance there."""
        # to account for incomplete evaluations from being promoted --> working on a copy
        observed_configs = (
            self.observed_configs.copy().dropna(inplace=False)
            if observed_configs is None
            else observed_configs
        )
        # remove the default from being part of a Successive-Halving bracket
        if (
            self.sample_default_first
            and self.sample_default_at_target
            and 0 in observed_configs.index.values
        ):
            observed_configs = observed_configs.drop(index=0)
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.clean_rung_information()
        for _rung in observed_configs.rung.unique():
            idxs = observed_configs.rung == _rung
            self.rung_members[_rung] = observed_configs.index[idxs].values
            self.rung_members_performance[_rung] = observed_configs.perf[idxs].values
        return

    def _handle_promotions(self):
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        self.rung_promotions = self.promotion_policy.retrieve_promotions()

    # pylint: disable=no-self-use
    def clear_old_brackets(self):
        return

    def _fit_models(self):
        # define any model or surrogate training and acquisition function state setting
        # if adding model-based search to the basic multi-fidelity algorithm
        return

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

        self.rung_histories = {
            rung: {"config": [], "perf": []}
            for rung in range(self.min_rung, self.max_rung + 1)
        }

        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(previous_results)
        self.total_fevals = len(previous_results) + len(pending_evaluations)

        # account for pending evaluations
        self._handle_pending_evaluations(pending_evaluations)

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # filter/reset old SH brackets
        self.clear_old_brackets()

        # identifying promotion list per rung
        self._handle_promotions()

        # fit any model/surrogates
        self._fit_models()

        return

    def is_init_phase(self) -> bool:
        return True

    def sample_new_config(
        self,
        rung: int = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # Samples configuration from policy or random
        if self.sampling_policy is None:
            config = self.pipeline_space.sample(
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )
        else:
            config = self.sampling_policy.sample(**self.sampling_args)
        return config

    def _generate_new_config_id(self):
        return self.observed_configs.index.max() + 1 if len(self.observed_configs) else 0

    def get_default_configuration(self):
        pass

    def is_promotable(self) -> int | None:
        """Returns an int if a rung can be promoted, else a None."""
        rung_to_promote = None

        # # iterates starting from the highest fidelity promotable to the lowest fidelity
        for rung in reversed(range(self.min_rung, self.max_rung)):
            if len(self.rung_promotions[rung]) > 0:
                rung_to_promote = rung
                # stop checking when a promotable config found
                # no need to search at lower fidelities
                break
        return rung_to_promote

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
            rung_id = self.min_rung
            # using random instead of np.random to be consistent with NePS BO
            if (
                self.use_priors
                and self.sample_default_first
                and len(self.observed_configs) == 0
            ):
                if self.sample_default_at_target:
                    # sets the default config to be evaluated at the target fidelity
                    rung_id = self.max_rung
                    self.logger.info("Next config will be evaluated at target fidelity.")
                self.logger.info("Sampling the default configuration...")
                config = self.pipeline_space.sample_default_configuration()

            elif random.random() < self.random_interleave_prob:
                config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=False,  # sample uniformly random
                    ignore_fidelity=True,
                )
            else:
                config = self.sample_new_config(rung=rung_id)

            fidelity_value = self.rung_map[rung_id]
            config.fidelity.value = fidelity_value

            previous_config_id = None
            config_id = f"{self._generate_new_config_id()}_{rung_id}"

        return config.hp_values(), config_id, previous_config_id  # type: ignore

    def _enhance_priors(self, confidence_score=None):
        """Only applicable when priors are given along with a confidence.

        Args:
            confidence_score: dict
                The confidence scores for the types.
                Example: {"categorical": 5.2, "numeric": 0.15}
        """
        if not self.use_priors and self.prior_confidence is None:
            return
        for k, v in self.pipeline_space.items():
            if v.is_fidelity or isinstance(v, ConstantParameter):
                continue
            elif isinstance(v, (FloatParameter, IntegerParameter)):
                if confidence_score is None:
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                else:
                    confidence = confidence_score["numeric"]
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(v, CategoricalParameter):
                if confidence_score is None:
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                else:
                    confidence = confidence_score["categorical"]
                self.pipeline_space[k].default_confidence_score = confidence


class SuccessiveHalving(SuccessiveHalvingBase):
    def _calc_budget_used_in_bracket(self, config_history: list[int]):
        budget = 0
        for rung in self.config_map.keys():
            count = sum(config_history == rung)
            # `range(min_rung, rung+1)` counts the black-box cost of promotions since
            # SH budgets assume each promotion involves evaluation from scratch
            budget += count * sum(np.arange(self.min_rung, rung + 1))
        return budget

    def clear_old_brackets(self):
        """Enforces reset at each new bracket.

        The _get_rungs_state() function creates the `rung_promotions` dict mapping which
        is used by the promotion policies to determine the next step: promotion/sample.
        The key to simulating reset of rungs like in vanilla SH is by subsetting only the
        relevant part of the observation history that corresponds to one SH bracket.
        Under a parallel run, multiple SH brackets can be spawned. The oldest, active,
        incomplete SH bracket is searched for to choose the next evaluation. If either
        all brackets are over or waiting, a new SH bracket is spawned.
        There are no waiting or blocking calls.
        """
        # indexes to mark separate brackets
        start = 0
        end = self.config_map[self.min_rung]  # length of lowest rung in a bracket
        if self.sample_default_at_target and self.sample_default_first:
            start += 1
            end += 1
        # iterates over the different SH brackets which span start-end by index
        while end <= len(self.observed_configs):
            # for the SH bracket in start-end, calculate total SH budget used
            bracket_budget_used = self._calc_budget_used_in_bracket(
                deepcopy(self.observed_configs.rung.values[start:end])
            )
            # if budget used is less than a SH bracket budget then still an active bracket
            if bracket_budget_used < sum(self.full_rung_trace):
                # subsetting only this SH bracket from the history
                self._get_rungs_state(self.observed_configs.iloc[start:end])
                # extra call to use the updated rung member info to find promotions
                # SyncPromotion signals a wait if a rung is full but with
                # incomplete/pending evaluations, and signals to starts a new SH bracket
                self._handle_promotions()
                promotion_count = 0
                for _, promotions in self.rung_promotions.items():
                    promotion_count += len(promotions)
                # if no promotion candidates are returned, then the current bracket
                # is active and waiting
                if promotion_count:
                    # returns the oldest active bracket if a promotion found
                    return
            # else move to next SH bracket recorded by an offset (= lowest rung length)
            start = end
            end = start + self.config_map[self.min_rung]

        # updates rung info with the latest active, incomplete bracket
        self._get_rungs_state(self.observed_configs.iloc[start:end])
        # _handle_promotion() need not be called as it is called by load_results()
        return


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
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",  # medium = 0.25
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
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
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
        )


class AsynchronousSuccessiveHalving(SuccessiveHalvingBase):
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
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
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
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
        )


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
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
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
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
        )


if __name__ == "__main__":
    pass
