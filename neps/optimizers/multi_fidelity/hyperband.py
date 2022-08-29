from __future__ import annotations

import typing

from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from .promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from .sampling_policy import FixedPriorPolicy, RandomUniformPolicy
from .successive_halving import SuccessiveHalving


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
        )
        super().__init__(**args)
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            # TODO update min budget for SH bracket
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = SuccessiveHalving(**args)
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))

    def _get_bracket_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        rung = self.full_rung_trace[self._counter]
        return rung

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        for s in range(self.max_rung + 1):
            self.sh_brackets[s].promotion_policy.set_state(
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
        current_SH_bracket = self._get_bracket_to_run()
        # update SH bracket with current state of promotion candidates
        config, config_id, previous_config_id = self.sh_brackets[
            current_SH_bracket
        ].get_config_and_ids()
        # IMPORTANT to tell SH to query the next allocation
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
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            initial_design_type=initial_design_type,
            use_priors=self.use_priors,  # key change to the base SH class
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
        )


class AsynchronousHyperband(Hyperband):
    """Implements ASHA but as Hyperband."""

    # could implement MOBSTER like sampling of SH bracket to run
    ...
