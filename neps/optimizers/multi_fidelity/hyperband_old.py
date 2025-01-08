from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.multi_fidelity.mf_bo import MFBOBase
from neps.optimizers.multi_fidelity.promotion_policy import (
    AsyncPromotionPolicy,
    SyncPromotionPolicy,
)
from neps.optimizers.multi_fidelity.sampling_policy import (
    ModelPolicy,
    RandomUniformPolicy,
)
from neps.sampling.priors import Prior
from neps.search_spaces import Categorical, Constant, Float, Integer
from neps.search_spaces.functions import sample_one_old

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions import (
        BaseAcquisition,
    )
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial
    from neps.utils.types import RawConfig

CUSTOM_FLOAT_CONFIDENCE_SCORES = dict(Float.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = dict(Categorical.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})


class HyperbandBase(BaseOptimizer):
    """Implements a Hyperband procedure with a sampling and promotion policy."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
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
        bracket_type: Literal["sync", "async"] = "sync",
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
        )
        if random_interleave_prob < 0 or random_interleave_prob > 1:
            raise ValueError("random_interleave_prob should be in [0.0, 1.0]")

        if bracket_type not in ["sync", "async"]:
            raise ValueError(
                "bracket_type should be either 'sync' or 'async'"
                f"but got {bracket_type}"
            )

        self.random_interleave_prob = random_interleave_prob
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        assert self.pipeline_space.fidelity is not None
        self.min_budget = self.pipeline_space.fidelity.lower
        self.max_budget = self.pipeline_space.fidelity.upper
        self.eta = eta
        self.sampling_policy = sampling_policy(pipeline_space=self.pipeline_space)
        self.promotion_policy = promotion_policy(self.eta)
        self.bracket_type = bracket_type

        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type
        self.use_priors = use_priors

        # check to ensure no rung ID is negative
        # equivalent to s_max in https://arxiv.org/pdf/1603.06560.pdf
        stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert stopping_rate_limit >= 0

        self.rung_map = {}
        nrungs = (
            int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta)))
            + 1
        )
        _max_budget = self.max_budget
        for i in reversed(range(nrungs)):
            self.rung_map[i] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, Integer)
                else _max_budget
            )
            _max_budget /= self.eta

        self.config_map: dict[int, int] = {}

        s_max = stopping_rate_limit + 1
        _s = stopping_rate_limit
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        for i in range(nrungs):
            self.config_map[i] = int(_n_config)
            _n_config //= self.eta

        self.min_rung = min(self.rung_map)
        self.max_rung = max(self.rung_map)

        # placeholder args for varying promotion and sampling policies
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

        self.rung_histories = {}

        sh_args = {
            "pipeline_space": pipeline_space,
            "budget": budget,
            "eta": eta,
            "initial_design_type": initial_design_type,
            "use_priors": use_priors,
            "sampling_policy": sampling_policy,
            "promotion_policy": promotion_policy,
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
            "ignore_errors": ignore_errors,
            "prior_confidence": prior_confidence,
            "random_interleave_prob": random_interleave_prob,
            "sample_default_first": sample_default_first,
            "sample_default_at_target": sample_default_at_target,
            "bracket_type": bracket_type,
        }

        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets: dict[int, BracketOptimizer] = {}
        for s in range(self.max_rung + 1):
            self.sh_brackets[s] = BracketOptimizer(early_stopping_rate=s, **sh_args)
            # `full_rung_trace` contains the index of SH bracket to run sequentially
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))

        # book-keeping variables
        self.current_sh_bracket: int = 0

    def clear_old_brackets(self) -> None:
        """Enforces reset at each new bracket."""
        # unlike synchronous SH, the state is not reset at each rung and a configuration
        # is promoted if the rung has eta configs if it is the top performing
        # base class allows for retaining the whole optimization state
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
        observed_configs, rung_histories = trials_to_table(
            trials=trials,
            space=self.pipeline_space,
            rung_range=(self.min_rung, self.max_rung),
            drop_config_id_0=self.sample_default_first and self.sample_default_at_target,
        )
        indices_of_highest_rung_per_config = observed_configs.groupby("id")[
            "rung"
        ].idxmax()
        observed_configs = observed_configs.loc[indices_of_highest_rung_per_config]
        print("---------")  # noqa: T201
        print(observed_configs, indices_of_highest_rung_per_config)  # noqa: T201
        self.observed_configs = observed_configs
        self.rung_histories = rung_histories

        # process optimization state and bucket observations per rung
        self.rung_members = {k: [] for k in self.rung_map}
        self.rung_members_performance = {k: [] for k in self.rung_map}
        self.rung_promotions = {k: [] for k in self.rung_map}
        obs_tmp = self.observed_configs.dropna(inplace=False)

        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        for _rung in obs_tmp["rung"].unique():
            idxs = obs_tmp["rung"] == _rung
            self.rung_members[_rung] = obs_tmp.index[idxs].values
            self.rung_members_performance[_rung] = obs_tmp.perf[idxs].values

        # filter/reset old SH brackets
        self.clear_old_brackets()

        # fit any model/surrogates
        self._fit_models()

        # important for the global HB to run the right SH
        bracket = self.sh_brackets[self.current_sh_bracket]
        bracket.observed_configs = self.observed_configs

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    @abstractmethod
    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        raise NotImplementedError

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


class Hyperband(HyperbandBase):
    def clear_old_brackets(self) -> None:
        """Enforces reset at each new bracket.

        The _get_rungs_state() function creates the `rung_promotions` dict mapping which
        is used by the promotion policies to determine the next step: promotion/sample.
        To simulate reset of rungs like in vanilla HB, the algorithm is viewed as a
        series of SH brackets, where the SH brackets comprising HB is repeated. This is
        done by iterating over the closed loop of possible SH brackets (self.sh_brackets).
        The oldest, active, incomplete SH bracket is searched for to choose the next
        evaluation. If either all brackets are over or waiting, a new SH bracket,
        corresponding to the SH bracket under HB as registered by `current_SH_bracket`.
        """
        n_sh_brackets = len(self.sh_brackets)
        # iterates over the different SH brackets
        self.current_sh_bracket = 0  # indexing from range(0, n_sh_brackets)
        start = 0
        _min_rung = self.sh_brackets[self.current_sh_bracket].min_rung
        end = self.sh_brackets[self.current_sh_bracket].rung_capacitires[_min_rung]

        # stores the base rung size for each SH bracket in HB
        base_rung_sizes = []  # sorted(self.config_map.values(), reverse=True)
        for bracket in self.sh_brackets.values():
            base_rung_sizes.append(
                sorted(bracket.rung_capacitires.values(), reverse=True)[0]
            )

        while end <= len(self.observed_configs):
            # subsetting only this SH bracket from the history
            sh_bracket = self.sh_brackets[self.current_sh_bracket]

            # > clean_rung_information()
            sh_bracket.rung_members = {k: [] for k in self.rung_map}
            sh_bracket.rung_members_performance = {k: [] for k in self.rung_map}
            sh_bracket.rung_promotions = {k: [] for k in self.rung_map}

            # for the SH bracket in start-end, calculate total SH budget used, from the
            # correct SH bracket object to make the right budget calculations

            assert isinstance(sh_bracket, BracketOptimizer)
            bracket_budget_used = tmp_calc_budget_used_in_bracket(
                rungs=sh_bracket.rung_capacitires.keys(),
                min_rung=sh_bracket.min_rung,
                config_history=self.observed_configs["rung"].to_numpy()[start:end],
            )
            # if budget used is less than the total SH budget then still an active bracket
            current_bracket_full_budget = sum(sh_bracket.full_rung_trace)
            if bracket_budget_used < current_bracket_full_budget:
                # updating rung information of the current bracket
                sh_bracket.rung_members = {k: [] for k in self.rung_map}
                sh_bracket.rung_members_performance = {k: [] for k in self.rung_map}
                sh_bracket.rung_promotions = {k: [] for k in self.rung_map}
                sh_obs_configs = self.observed_configs.iloc[start:end]
                for _rung in sh_obs_configs["rung"].unique():
                    idxs = sh_obs_configs["rung"] == _rung
                    sh_bracket.rung_members[_rung] = sh_obs_configs.index[idxs].values
                    sh_bracket.rung_members_performance[_rung] = sh_obs_configs.perf[
                        idxs
                    ].values

                # extra call to use the updated rung member info to find promotions
                # SyncPromotion signals a wait if a rung is full but with
                # incomplete/pending evaluations, signals to starts a new SH bracket
                sh_bracket.rung_promotions = (
                    sh_bracket.promotion_policy.retrieve_promotions(
                        max_rung=sh_bracket.max_rung,
                        rung_members=sh_bracket.rung_members,
                        rung_members_performance=sh_bracket.rung_members_performance,
                        config_map=sh_bracket.rung_capacitires,
                    )
                )

                promotion_count = 0
                for _, promotions in sh_bracket.rung_promotions.items():
                    promotion_count += len(promotions)
                # if no promotion candidates are returned, then the current bracket
                # is active and waiting
                if promotion_count:
                    # returns the oldest active bracket if a promotion found which is the
                    # current SH bracket at this scope
                    return
                # if no promotions, ensure an empty state explicitly to disable bracket
                # > clean_rung_information()
                sh_bracket.rung_members = {k: [] for k in self.rung_map}
                sh_bracket.rung_members_performance = {k: [] for k in self.rung_map}
                sh_bracket.rung_promotions = {k: [] for k in self.rung_map}

            start = end
            # updating pointer to the next SH bracket in HB
            self.current_sh_bracket = (self.current_sh_bracket + 1) % n_sh_brackets
            end = start + base_rung_sizes[self.current_sh_bracket]
        # reaches here if all old brackets are either waiting or finished

        # updates rung info with the latest active, incomplete bracket
        sh_bracket = self.sh_brackets[self.current_sh_bracket]
        sh_bracket.rung_members = {k: [] for k in self.rung_map}
        sh_bracket.rung_members_performance = {k: [] for k in self.rung_map}
        sh_bracket.rung_promotions = {k: [] for k in self.rung_map}
        sh_obs_configs = self.observed_configs.iloc[start:end]
        for _rung in sh_obs_configs["rung"].unique():
            idxs = sh_obs_configs["rung"] == _rung
            sh_bracket.rung_members[_rung] = sh_obs_configs.index[idxs].values
            sh_bracket.rung_members_performance[_rung] = sh_obs_configs.perf[idxs].values
        sh_bracket.rung_promotions = sh_bracket.promotion_policy.retrieve_promotions(
            max_rung=sh_bracket.max_rung,
            rung_members=sh_bracket.rung_members,
            rung_members_performance=sh_bracket.rung_members_performance,
            config_map=sh_bracket.rung_capacitires,
        )

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        config, config_id, previous_config_id = self.sh_brackets[
            self.current_sh_bracket
        ].get_config_and_ids()
        return config, config_id, previous_config_id


class AsynchronousHyperband(HyperbandBase):
    """Implements ASHA but as Hyperband.

    Implements the Promotion variant of ASHA as used in Mobster.
    """

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: Any = RandomUniformPolicy,
        promotion_policy: Any = AsyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] | None = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
    ):
        args = {
            "pipeline_space": pipeline_space,
            "budget": budget,
            "eta": eta,
            "initial_design_type": initial_design_type,
            "use_priors": use_priors,
            "sampling_policy": sampling_policy,
            "promotion_policy": promotion_policy,
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
            "ignore_errors": ignore_errors,
            "prior_confidence": prior_confidence,
            "random_interleave_prob": random_interleave_prob,
            "sample_default_first": sample_default_first,
            "sample_default_at_target": sample_default_at_target,
        }
        super().__init__(**args)
        # overwrite parent class SH brackets with Async SH brackets
        self.sh_brackets: dict[int, BracketOptimizer] = {}
        for s in range(self.max_rung + 1):
            self.sh_brackets[s] = BracketOptimizer(early_stopping_rate=s, **args)

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = sample_bracket_to_run(max_rung=self.max_rung, eta=self.eta)
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id


class MOBSTER(MFBOBase, AsynchronousHyperband):
    model_based = True
    modelling_type = "rung"

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: Any = RandomUniformPolicy,
        promotion_policy: Any = AsyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] | None = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        # new arguments for model
        model_policy: Any = ModelPolicy,
        surrogate_model: str | Any = "gp",  # TODO: Remove
        domain_se_kernel: str | None = None,  # TODO: Remove
        hp_kernels: list | None = None,  # TODO: Remove
        surrogate_model_args: dict | None = None,  # TODO: Remove
        acquisition: str | BaseAcquisition = "EI",  # TODO: Remove
        log_prior_weighted: bool = False,  # TODO: Remove
        acquisition_sampler: str = "random",  # TODO: Remove
    ):
        hb_args = {
            "pipeline_space": pipeline_space,
            "budget": budget,
            "eta": eta,
            "initial_design_type": initial_design_type,
            "use_priors": use_priors,
            "sampling_policy": sampling_policy,
            "promotion_policy": promotion_policy,
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
            "ignore_errors": ignore_errors,
            "prior_confidence": prior_confidence,
            "random_interleave_prob": random_interleave_prob,
            "sample_default_first": sample_default_first,
            "sample_default_at_target": sample_default_at_target,
        }
        super().__init__(**hb_args)

        self.pipeline_space.has_prior = self.use_priors

        # counting non-fidelity dimensions in search space
        ndims = sum(
            1
            for _, hp in self.pipeline_space.hyperparameters.items()
            if not hp.is_fidelity
        )
        n_min = ndims + 1
        self.init_size = n_min + 1  # in BOHB: init_design >= N_min + 2

        if self.use_priors:
            prior = Prior.from_space(self.pipeline_space, include_fidelity=False)
        else:
            prior = None

        self.model_policy = model_policy(pipeline_space=pipeline_space, prior=prior)

        for _, sh in self.sh_brackets.items():
            sh.model_policy = self.model_policy  # type: ignore
            sh.sample_new_config = self.sample_new_config  # type: ignore


# TODO: TrulyAsyncHyperband
