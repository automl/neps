from __future__ import annotations

import typing

import numpy as np
from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import FixedPriorPolicy, RandomUniformPolicy
from .successive_halving import AsynchronousSuccessiveHalving, SuccessiveHalving


class HyperbandBase(SuccessiveHalving):
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
        sample_default_first: bool = False,
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
            sample_default_first=sample_default_first,
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
        # book-keeping variables
        self.current_sh_bracket = None
        self.old_history_len = None

    def _get_bracket_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        bracket = self.full_rung_trace[self._counter % len(self.full_rung_trace)]
        return bracket

    def _update_state_counter(self) -> None:
        # TODO: get rid of this dependency
        self._counter += 1

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the current SH bracket needs the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        s = self.current_sh_bracket
        self.sh_brackets[s].promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            config_map=self.sh_brackets[s].config_map,
        )
        self.sh_brackets[s].rung_promotions = self.sh_brackets[
            s
        ].promotion_policy.retrieve_promotions()
        self.sh_brackets[s].observed_configs = self.observed_configs.copy()

    def _retrieve_current_state(self) -> tuple[int, int]:
        """Returns the current SH bracket number and the ordered index of config IDs
        that are history to ignore.
        """
        nsamples_per_bracket = sum(
            v.config_map[v.min_rung] for v in self.sh_brackets.values()
        )
        nevals_per_bracket = len(self.full_rung_trace)
        # TODO: remove dependency on `_counter` by calculating it from `observed_configs`
        nbrackets = self._counter // nevals_per_bracket
        old_history_len = nbrackets * nsamples_per_bracket
        # current HB bracket configs
        new_bracket_len = self._counter % nevals_per_bracket
        history_offset = 0
        for s in sorted(self.sh_brackets.keys()):
            _sh_bracket = self.sh_brackets[s]
            sh_bracket_len = len(_sh_bracket.full_rung_trace)
            if new_bracket_len >= sh_bracket_len:
                history_offset += _sh_bracket.config_map[_sh_bracket.min_rung]
                new_bracket_len -= sh_bracket_len
            else:
                break
        old_history_len += history_offset
        return s, old_history_len

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
        # the current SH bracket to execute in the current HB iteration
        current_sh_bracket = self._get_bracket_to_run()
        # update SH bracket with current state of promotion candidates
        config, config_id, previous_config_id = self.sh_brackets[
            current_sh_bracket
        ].get_config_and_ids()
        # IMPORTANT function call to tell synchronous SH to query the next allocation
        self._update_state_counter()
        return config, config_id, previous_config_id  # type: ignore


class Hyperband(HyperbandBase):
    def clear_old_brackets(self):
        """Enforces reset at each new bracket."""
        # overloaded from SH since HB needs to not only forget the full past HB brackets
        # but also the previous SH brackets in the current HB bracket
        self.current_sh_bracket, self.old_history_len = self._retrieve_current_state()
        self.config_map = self.sh_brackets[self.current_sh_bracket].config_map
        if self.old_history_len > 0:
            # disregarding older HB brackets + older SH brackets in current HB bracket
            self._get_rungs_state(self.observed_configs.loc[self.old_history_len :])
        return


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
        sample_default_first: bool = False,
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
            sample_default_first=sample_default_first,
        )


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
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = False,
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
            sample_default_first=sample_default_first,
        )
        super().__init__(**args)
        # overwrite parent class SH brackets with Async SH brackets
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = AsynchronousSuccessiveHalving(**args)

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        # s = self.current_sh_bracket
        for s, bracket in self.sh_brackets.items():
            bracket.promotion_policy.set_state(
                max_rung=self.max_rung,
                members=self.rung_members,
                performances=self.rung_members_performance,
                config_map=bracket.config_map,
            )
            bracket.rung_promotions = bracket.promotion_policy.retrieve_promotions()
            bracket.observed_configs = self.observed_configs.copy()

    def _get_bracket_to_run(self):
        """Samples the ASHA bracket to run"""
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
        sample_default_first: bool = False,
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
            sample_default_first=sample_default_first,
        )
