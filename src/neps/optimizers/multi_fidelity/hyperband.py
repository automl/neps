# type: ignore

from __future__ import annotations

import typing
from copy import deepcopy
from typing import Any

import numpy as np
from typing_extensions import Literal

from metahyper import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..bayesian_optimization.acquisition_functions.base_acquisition import BaseAcquisition
from ..bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from .mf_bo import MFBOBase
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import (
    EnsemblePolicy,
    FixedPriorPolicy,
    ModelPolicy,
    RandomUniformPolicy,
)
from .successive_halving import (
    AsynchronousSuccessiveHalving,
    SuccessiveHalving,
    SuccessiveHalvingBase,
)


class HyperbandBase(SuccessiveHalvingBase):
    """Implements a Hyperband procedure with a sampling and promotion policy."""

    early_stopping_rate = 0

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
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
        args = dict(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=self.early_stopping_rate,  # HB subsumes this param of SH
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
        super().__init__(**args)
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = SuccessiveHalving(**args)
            # `full_rung_trace` contains the index of SH bracket to run sequentially
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))
        # book-keeping variables
        self.current_sh_bracket = None  # type: ignore
        self.old_history_len = None

    def _update_state_counter(self) -> None:
        # TODO: get rid of this dependency
        self._counter += 1

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the current SH bracket needs the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        # `clean_active_brackets` takes care of setting rung information and promotion
        # for the current SH bracket in HB
        # TODO: can we avoid copying full observation history
        bracket = self.sh_brackets[self.current_sh_bracket]  # type: ignore
        bracket.observed_configs = self.observed_configs.copy()

    # pylint: disable=no-self-use
    def clear_old_brackets(self):
        """Enforces reset at each new bracket."""
        # unlike synchronous SH, the state is not reset at each rung and a configuration
        # is promoted if the rung has eta configs if it is the top performing
        # base class allows for retaining the whole optimization state
        return

    def _handle_promotions(self):
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        # promotions are handled by the individual SH brackets which are explicitly
        # called in the _update_sh_bracket_state() function
        # overloaded function disables the need for retrieving promotions for HB overall
        return

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        super().load_results(previous_results, pending_evaluations)
        # important for the global HB to run the right SH
        self._update_sh_bracket_state()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        raise NotImplementedError


class Hyperband(HyperbandBase):
    def clear_old_brackets(self):
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
            # pylint: disable=protected-access
            bracket_budget_used = sh_bracket._calc_budget_used_in_bracket(
                deepcopy(self.observed_configs.rung.values[start:end])
            )
            # if budget used is less than the total SH budget then still an active bracket
            current_bracket_full_budget = sum(sh_bracket.full_rung_trace)
            if bracket_budget_used < current_bracket_full_budget:
                # updating rung information of the current bracket
                # pylint: disable=protected-access
                sh_bracket._get_rungs_state(self.observed_configs.iloc[start:end])
                # extra call to use the updated rung member info to find promotions
                # SyncPromotion signals a wait if a rung is full but with
                # incomplete/pending evaluations, signals to starts a new SH bracket
                sh_bracket._handle_promotions()  # pylint: disable=protected-access
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
        # pylint: disable=protected-access
        sh_bracket._get_rungs_state(self.observed_configs.iloc[start:end])
        sh_bracket._handle_promotions()
        # self._handle_promotion() need not be called as it is called by load_results()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        config, config_id, previous_config_id = self.sh_brackets[
            self.current_sh_bracket  # type: ignore
        ].get_config_and_ids()
        return config, config_id, previous_config_id


class HyperbandWithPriors(Hyperband):
    """Implements a Hyperband procedure with a sampling and promotion policy."""

    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
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
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,  # key change to the base HB class
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


class HyperbandCustomDefault(HyperbandWithPriors):
    """If prior specified, does 50% times priors and 50% random search like vanilla-HB."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
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
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target
        )
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 0.5,
                "inc": 0,
                "random": 0.5,
            },
        }
        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args


class AsynchronousHyperband(HyperbandBase):
    """Implements ASHA but as Hyperband.

    Implements the Promotion variant of ASHA as used in Mobster.
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RandomUniformPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
    ):
        args = dict(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
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
        super().__init__(**args)
        # overwrite parent class SH brackets with Async SH brackets
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            # key difference from vanilla HB where it runs synchronous SH brackets
            self.sh_brackets[s] = AsynchronousSuccessiveHalving(**args)

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

    def _get_bracket_to_run(self):
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
        bracket_next = np.random.choice(range(self.max_rung + 1), p=bracket_probs)
        return bracket_next

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = self._get_bracket_to_run()
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id  # type: ignore


class AsynchronousHyperbandWithPriors(AsynchronousHyperband):
    """Implements ASHA but as Hyperband."""

    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
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
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,  # key change to the base Async HB class
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


class MOBSTER(MFBOBase, AsynchronousHyperband):
    model_based = True
    modelling_type = "rung"

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RandomUniformPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        # new arguments for model
        model_policy: typing.Any = ModelPolicy,
        surrogate_model: str | Any = "gp",
        domain_se_kernel: str = None,
        hp_kernels: list = None,
        surrogate_model_args: dict = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "random",
    ):
        hb_args = dict(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
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
        super().__init__(**hb_args)

        self.pipeline_space.has_prior = self.use_priors

        bo_args = dict(
            surrogate_model=surrogate_model,
            domain_se_kernel=domain_se_kernel,
            hp_kernels=hp_kernels,
            surrogate_model_args=surrogate_model_args,
            acquisition=acquisition,
            log_prior_weighted=log_prior_weighted,
            acquisition_sampler=acquisition_sampler,
        )
        # counting non-fidelity dimensions in search space
        ndims = sum(
            1
            for _, hp in self.pipeline_space.hyperparameters.items()
            if not hp.is_fidelity
        )
        n_min = ndims + 1
        self.init_size = n_min + 1  # in BOHB: init_design >= N_min + 2
        self.model_policy = model_policy(pipeline_space, **bo_args)

        for _, sh in self.sh_brackets.items():
            sh.model_policy = self.model_policy
            sh.sample_new_config = self.sample_new_config


# TODO: TrulyAsyncHyperband
