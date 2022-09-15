from __future__ import annotations

import typing

import numpy as np
from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import FixedPriorPolicy, RandomUniformPolicy
from .successive_halving import AsynchronousSuccessiveHalving, SuccessiveHalving


class Hyperband(SuccessiveHalving):
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
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        super().__init__(**args)
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = SuccessiveHalving(**args)
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))

    def _get_bracket_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        bracket = self.full_rung_trace[self._counter]
        return bracket

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        for s in range(self.max_rung + 1):
            self.sh_brackets[s].promotion_policy.set_state(
                max_rung=self.max_rung,
                members=self.rung_members,
                performances=self.rung_members_performance,
                **self.promotion_policy_kwargs,
            )
            self.sh_brackets[
                s
            ].rung_promotions = self.promotion_policy.retrieve_promotions()
            self.sh_brackets[s].observed_configs = self.observed_configs.copy()

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        super().load_results(previous_results, pending_evaluations)
        self._update_sh_bracket_state()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the current SH bracket to execute in the current HB iteration
        current_sh_bracket = self._get_bracket_to_run()
        # update SH bracket with current state of promotion candidates
        config, config_id, previous_config_id = self.sh_brackets[
            current_sh_bracket
        ].get_config_and_ids()
        # IMPORTANT to tell synchronous SH to query the next allocation
        self._update_state_counter()
        return config, config_id, previous_config_id  # type: ignore


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
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )


class AsynchronousHyperband(Hyperband):
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
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        super().__init__(**args)
        # overwrite parent class SH brackets with Async SH brackets
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = AsynchronousSuccessiveHalving(**args)

    def _get_bracket_to_run(self):
        """Samples the ASHA bracket to run"""
        # Sampling distribution from Appendix A in https://arxiv.org/abs/2003.10865
        K = 5
        bracket_probs = [
            self.eta ** (K - s) * (K + 1) / (K - s + 1) for s in range(self.max_rung + 1)
        ]
        bracket_probs = np.array(bracket_probs) / sum(bracket_probs)
        bracket_next = np.random.choice(range(self.max_rung + 1), p=bracket_probs)
        return bracket_next


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
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
