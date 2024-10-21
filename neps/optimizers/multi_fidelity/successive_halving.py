from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.multi_fidelity.promotion_policy import (
    SyncPromotionPolicy,
)
from neps.optimizers.multi_fidelity.sampling_policy import (
    RandomUniformPolicy,
)
from neps.search_spaces import (
    Categorical,
    Constant,
    Float,
    Integer,
    SearchSpace,
)
from neps.search_spaces.functions import sample_one_old

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial
    from neps.utils.types import RawConfig

logger = logging.getLogger(__name__)

CUSTOM_FLOAT_CONFIDENCE_SCORES = dict(Float.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = dict(Categorical.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})

# We should replace this with some class
RUNG_HISTORY_TYPE = dict[int, dict[Literal["config", "perf"], list[int | float]]]


# TODO: Remove this, just using it in the meantime to delete methods from the mf
# optimizers
def tmp_load_results(
    trials: Mapping[str, Trial],
    space: SearchSpace,
    rung_range: tuple[int, int],
) -> tuple[pd.DataFrame, RUNG_HISTORY_TYPE]:
    min_rung, max_rung = rung_range

    rung_histories: RUNG_HISTORY_TYPE = {
        rung: {"config": [], "perf": []} for rung in range(min_rung, max_rung + 1)
    }
    records: dict[int, dict] = {}

    for trial_id, trial in trials.items():
        config_id_str, rung_str = trial_id.split("_")
        _config_id, _rung = int(config_id_str), int(rung_str)

        if trial.report is None:
            perf = np.nan  # Pending
        elif trial.report.loss is None:
            perf = np.inf  # Error? Either way, we wont promote it
        else:
            perf = trial.report.loss

        # If there has but none previously or the new one is more
        # up to date, insert it in.
        previous_seen = records.get(_config_id)
        if previous_seen is None or _rung > previous_seen["rung"]:
            records[_config_id] = {
                "config": space.from_dict(trial.config),
                "rung": _rung,
                "perf": perf,
            }

        rung_histories[_rung]["config"].append(_config_id)
        rung_histories[_rung]["perf"].append(perf)

    _df = (
        pd.DataFrame.from_dict(
            records,
            orient="index",
            columns=["config", "rung", "perf"],  # type: ignore
        )
        .rename_axis("id")
        .astype({"config": object, "rung": int, "perf": float})
        .sort_index()
    )
    return _df, rung_histories


class SuccessiveHalvingBase(BaseOptimizer):
    """Implements a SuccessiveHalving procedure with a sampling and promotion policy."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int | None = None,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: Any = RandomUniformPolicy,
        promotion_policy: Any = SyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] | None = None,
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
        )
        if random_interleave_prob < 0 or random_interleave_prob > 1:
            raise ValueError("random_interleave_prob should be in [0.0, 1.0]")
        self.random_interleave_prob = random_interleave_prob
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        assert self.pipeline_space.fidelity is not None
        self.min_budget = self.pipeline_space.fidelity.lower
        self.max_budget = self.pipeline_space.fidelity.upper
        self.eta = eta
        # SH implicitly sets early_stopping_rate to 0
        # the parameter is exposed to allow HB to call SH with different stopping rates
        self.early_stopping_rate = early_stopping_rate
        self.sampling_policy = sampling_policy(pipeline_space=self.pipeline_space)
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
        self.rung_map = {}
        assert self.early_stopping_rate <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**self.early_stopping_rate)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        _max_budget = self.max_budget
        for i in reversed(range(nrungs)):
            self.rung_map[i + self.early_stopping_rate] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, Integer)
                else _max_budget
            )
            _max_budget /= self.eta

        self.config_map = {}
        new_min_budget = self.min_budget * (self.eta**self.early_stopping_rate)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        s_max = self.stopping_rate_limit + 1
        _s = self.stopping_rate_limit - self.early_stopping_rate
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        for i in range(nrungs):
            self.config_map[i + self.early_stopping_rate] = int(_n_config)
            _n_config //= self.eta

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
        self.rung_members_performance: dict = {}  # performances recorded per rung
        self.rung_promotions: dict = {}  # records a promotable config per rung

        # setup SH state counter
        self.full_rung_trace = []
        for rung in sorted(self.rung_map.keys()):
            self.full_rung_trace.extend([rung] * self.config_map[rung])

        #############################
        # Setting prior confidences #
        #############################
        # the std. dev or peakiness of distribution
        self.prior_confidence = prior_confidence
        if self.use_priors and self.prior_confidence is not None:
            for k, v in self.pipeline_space.items():
                if v.is_fidelity or isinstance(v, Constant):
                    continue
                if isinstance(v, Float | Integer):
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                    self.pipeline_space[k].default_confidence_score = confidence
                elif isinstance(v, Categorical):
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                    self.pipeline_space[k].default_confidence_score = confidence

        self.rung_histories: RUNG_HISTORY_TYPE = {}

    def clean_rung_information(self) -> None:
        self.rung_members = {k: [] for k in self.rung_map}
        self.rung_members_performance = {k: [] for k in self.rung_map}
        self.rung_promotions = {k: [] for k in self.rung_map}

    def _get_rungs_state(self, observed_configs: pd.DataFrame | None = None) -> None:
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

    def _handle_promotions(self) -> None:
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        self.rung_promotions = self.promotion_policy.retrieve_promotions()

    def clear_old_brackets(self) -> None:
        return

    def _fit_models(self) -> None:
        # define any model or surrogate training and acquisition function state setting
        # if adding model-based search to the basic multi-fidelity algorithm
        return

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        """This is basically the fit method."""
        observed_configs, rung_histories = tmp_load_results(
            trials=trials,
            space=self.pipeline_space,
            rung_range=(self.min_rung, self.max_rung),
        )
        self.observed_configs = observed_configs
        self.rung_histories = rung_histories

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # filter/reset old SH brackets
        self.clear_old_brackets()

        # identifying promotion list per rung
        self._handle_promotions()

        # fit any model/surrogates
        self._fit_models()

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    def sample_new_config(
        self,
        rung: int | None = None,
        **kwargs: Any,
    ) -> SearchSpace:
        # Samples configuration from policy or random
        if self.sampling_policy is None:
            return sample_one_old(
                self.pipeline_space,
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )

        return self.sampling_policy.sample(**self.sampling_args)

    def is_promotable(self) -> int | None:
        """Returns an int if a rung can be promoted, else a None."""
        # Find the first promotable config starting from the max rung
        for rung in reversed(range(self.min_rung, self.max_rung)):
            if len(self.rung_promotions[rung]) > 0:
                return rung

        return None

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        fidelity_name = self.pipeline_space.fidelity_name
        assert fidelity_name is not None

        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promotes the first recorded promotable config in the argsort-ed rung
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = row["config"].clone()
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at

            config_values = config._values
            config_values[fidelity_name] = self.rung_map[rung]

            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
        else:
            rung_id = self.min_rung
            # using random instead of np.random to be consistent with NePS BO
            rng = random.Random(None)  # TODO: Seeding
            if (
                self.use_priors
                and self.sample_default_first
                and len(self.observed_configs) == 0
            ):
                if self.sample_default_at_target:
                    # sets the default config to be evaluated at the target fidelity
                    rung_id = self.max_rung
                    logger.info("Next config will be evaluated at target fidelity.")
                logger.info("Sampling the default configuration...")
                config = self.pipeline_space.from_dict(self.pipeline_space.default_config)
            elif rng.random() < self.random_interleave_prob:
                config = sample_one_old(
                    self.pipeline_space,
                    patience=self.patience,
                    user_priors=False,  # sample uniformly random
                    ignore_fidelity=True,
                )
            else:
                config = self.sample_new_config(rung=rung_id)

            fidelity_value = self.rung_map[rung_id]
            config_values = config._values
            config_values[fidelity_name] = fidelity_value

            previous_config_id = None

            if len(self.observed_configs) == 0:
                new_config_id = 0
            else:
                _max = int(self.observed_configs.index.max())  # type: ignore
                new_config_id = _max + 1

            config_id = f"{new_config_id}_{rung_id}"

        return config_values, config_id, previous_config_id


class SuccessiveHalving(SuccessiveHalvingBase):
    def _calc_budget_used_in_bracket(self, config_history: list[int]) -> int:
        budget = 0
        for rung in self.config_map:
            count = sum(config_history == rung)
            # `range(min_rung, rung+1)` counts the black-box cost of promotions since
            # SH budgets assume each promotion involves evaluation from scratch
            budget += count * sum(np.arange(self.min_rung, rung + 1))
        return budget

    def clear_old_brackets(self) -> None:
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

            # TODO(eddiebergman): Not idea what the type is of the stuff in the deepcopy
            # but should work on removing the deepcopy
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
