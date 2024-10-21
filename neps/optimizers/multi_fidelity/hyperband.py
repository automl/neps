from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import SampledConfig
from neps.optimizers.multi_fidelity.mf_bo import MFBOBase
from neps.optimizers.multi_fidelity.promotion_policy import (
    AsyncPromotionPolicy,
    SyncPromotionPolicy,
)
from neps.optimizers.multi_fidelity.sampling_policy import (
    FixedPriorPolicy,
    ModelPolicy,
    RandomUniformPolicy,
)
from neps.optimizers.multi_fidelity.successive_halving import (
    SuccessiveHalving,
    SuccessiveHalvingBase,
)
from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions import (
        BaseAcquisition,
    )
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial
    from neps.utils.types import ConfigResult, RawConfig


class HyperbandBase(SuccessiveHalvingBase):
    """Implements a Hyperband procedure with a sampling and promotion policy."""

    early_stopping_rate = 0

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
    ):
        args = {
            "pipeline_space": pipeline_space,
            "budget": budget,
            "eta": eta,
            "early_stopping_rate": self.early_stopping_rate,  # HB subsumes this from SH
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
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets: dict[int, SuccessiveHalvingBase] = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = SuccessiveHalving(**args)
            # `full_rung_trace` contains the index of SH bracket to run sequentially
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))
        # book-keeping variables
        self.current_sh_bracket: int = 0

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the current SH bracket needs the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        # `clean_active_brackets` takes care of setting rung information and promotion
        # for the current SH bracket in HB
        # TODO: can we avoid copying full observation history
        bracket = self.sh_brackets[self.current_sh_bracket]
        bracket.observed_configs = self.observed_configs.copy()

    def clear_old_brackets(self) -> None:
        """Enforces reset at each new bracket."""
        # unlike synchronous SH, the state is not reset at each rung and a configuration
        # is promoted if the rung has eta configs if it is the top performing
        # base class allows for retaining the whole optimization state
        return

    def _handle_promotions(self) -> None:
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        # promotions are handled by the individual SH brackets which are explicitly
        # called in the _update_sh_bracket_state() function
        # overloaded function disables the need for retrieving promotions for HB overall

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        completed: dict[str, ConfigResult] = {
            trial_id: trial.into_config_result(self.pipeline_space.from_dict)
            for trial_id, trial in trials.items()
            if trial.report is not None
        }
        pending: dict[str, SearchSpace] = {
            trial_id: self.pipeline_space.from_dict(trial.config)
            for trial_id, trial in trials.items()
            if trial.report is None
        }

        self.rung_histories = {
            rung: {"config": [], "perf": []}
            for rung in range(self.min_rung, self.max_rung + 1)
        }

        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(completed)

        # account for pending evaluations
        self._handle_pending_evaluations(pending)

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # filter/reset old SH brackets
        self.clear_old_brackets()

        # identifying promotion list per rung
        self._handle_promotions()

        # fit any model/surrogates
        self._fit_models()

        # important for the global HB to run the right SH
        self._update_sh_bracket_state()

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    @abstractmethod
    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        raise NotImplementedError


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
        end = self.sh_brackets[self.current_sh_bracket].config_map[_min_rung]

        if self.sample_default_first and self.sample_default_at_target:
            start += 1
            end += 1

        # stores the base rung size for each SH bracket in HB
        base_rung_sizes = []  # sorted(self.config_map.values(), reverse=True)
        for bracket in self.sh_brackets.values():
            base_rung_sizes.append(sorted(bracket.config_map.values(), reverse=True)[0])
        while end <= len(self.observed_configs):
            # subsetting only this SH bracket from the history
            sh_bracket = self.sh_brackets[self.current_sh_bracket]
            sh_bracket.clean_rung_information()
            # for the SH bracket in start-end, calculate total SH budget used, from the
            # correct SH bracket object to make the right budget calculations

            assert isinstance(sh_bracket, SuccessiveHalving)
            bracket_budget_used = sh_bracket._calc_budget_used_in_bracket(
                deepcopy(self.observed_configs.rung.values[start:end])
            )
            # if budget used is less than the total SH budget then still an active bracket
            current_bracket_full_budget = sum(sh_bracket.full_rung_trace)
            if bracket_budget_used < current_bracket_full_budget:
                # updating rung information of the current bracket

                sh_bracket._get_rungs_state(self.observed_configs.iloc[start:end])
                # extra call to use the updated rung member info to find promotions
                # SyncPromotion signals a wait if a rung is full but with
                # incomplete/pending evaluations, signals to starts a new SH bracket
                sh_bracket._handle_promotions()
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
                sh_bracket.clean_rung_information()
            start = end
            # updating pointer to the next SH bracket in HB
            self.current_sh_bracket = (self.current_sh_bracket + 1) % n_sh_brackets
            end = start + base_rung_sizes[self.current_sh_bracket]
        # reaches here if all old brackets are either waiting or finished

        # updates rung info with the latest active, incomplete bracket
        sh_bracket = self.sh_brackets[self.current_sh_bracket]

        sh_bracket._get_rungs_state(self.observed_configs.iloc[start:end])
        sh_bracket._handle_promotions()
        # self._handle_promotion() need not be called as it is called by load_results()

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
        self.sh_brackets: dict[int, SuccessiveHalvingBase] = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = SuccessiveHalvingBase(**args)

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        for _, bracket in self.sh_brackets.items():
            bracket.promotion_policy.set_state(
                max_rung=self.max_rung,
                members=self.rung_members,
                performances=self.rung_members_performance,
                config_map=bracket.config_map,
            )
            bracket.rung_promotions = bracket.promotion_policy.retrieve_promotions()
            bracket.observed_configs = self.observed_configs.copy()

    def _get_bracket_to_run(self) -> int:
        """Samples the ASHA bracket to run.

        The selected bracket always samples at its minimum rung. Thus, selecting a bracket
        effectively selects the rung that a new sample will be evaluated at.
        """
        # Sampling distribution derived from Appendix A (https://arxiv.org/abs/2003.10865)
        # Adapting the distribution based on the current optimization state
        # s \in [0, max_rung] and to with the denominator's constraint, we have K > s - 1
        # and thus K \in [1, ..., max_rung, ...]
        # Since in this version, we see the full SH rung, we fix the K to max_rung
        K = self.max_rung
        bracket_probs = [
            self.eta ** (K - s) * (K + 1) / (K - s + 1) for s in range(self.max_rung + 1)
        ]
        bracket_probs = np.array(bracket_probs) / sum(bracket_probs)
        return int(np.random.choice(range(self.max_rung + 1), p=bracket_probs))

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = self._get_bracket_to_run()
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id


class AsynchronousHyperbandWithPriors(AsynchronousHyperband):
    """Implements ASHA but as Hyperband."""

    use_priors = True

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: Any = FixedPriorPolicy,
        promotion_policy: Any = AsyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,  # key change to the base Async HB class
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
        )


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
