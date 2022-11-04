import typing

import numpy as np
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalvingWithPriors


class OurOptimizerV1(AsynchronousSuccessiveHalvingWithPriors):
    """Implements a Prior Weighted Promotion Policy over ASHA.

    Performs promotions by ranking configs by their performances weighted by prior-score.
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
        )

    def find_rung_visits(self) -> dict:
        """Finds the unique number of configurations each rung has seen so far."""
        rung_visits = dict()
        for rung in range(self.max_rung + 1):
            # `observed_configs` stores the lates rung of each config evaluated
            # a config will be counted for all rungs including and below its current rung
            #
            # NOTE: This logic will fail for Hyperband as no information available on
            #   which rung was the config was evaluated first
            #
            rung_idx = np.where(self.observed_configs.rung.values >= rung)[0]
            rung_visits[rung] = len(rung_idx)
        return rung_visits

    def calc_decay(self, score: float, rung: int) -> float:
        """Decays the prior score like PiBO but with a multi-fidelity decay.

        Pi(x) ^ (Beta / n_z); where n_z is the number of configurations seen in
        optimization history that has been evaluated at the fidelity z
        """
        beta = 4  # from PiBO paper
        rung_visits = self.find_rung_visits()  # contains the config count per fidelity
        decay_rate = score ** (beta / (rung_visits[rung] + 1))
        return decay_rate

    def _get_rungs_state(self):
        super()._get_rungs_state()
        rung_weighted_performance = dict()
        y_star = self.get_incumbent_score()
        for rung in self.rung_members.keys():
            # multiplying the performance of each config with the prior score for it
            # equivalent to p(x) times y(x), where x is a config at a rung
            # and y(x) is normalized by y*
            rung_weighted_performance[rung] = [
                v
                * self.calc_decay(
                    self.observed_configs.config[
                        self.rung_members[rung][i]
                    ].compute_prior(),
                    rung,
                )
                / y_star
                for i, v in enumerate(self.rung_members_performance[rung])
            ]
        self.rung_members_performance = rung_weighted_performance
        return


class OurOptimizerV1_2(OurOptimizerV1):
    def calc_decay(self, score: float, rung: int) -> float:
        """Decays the prior score like PiBO but with a multi-fidelity decay.

        Pi(x) ^ (Beta * (z_max - z) / n); where n is the number of unique configurations
        seen in optimization history. z is the current fidelity rung level and z_max is
        the target fidelity. The exponent here is the same as PiBO but scaled by the
        difference in rung levels or discrete fidelity levels.

        NOTE: the denominator is now the total unique number of observations and
          not per fidelity observations
        """
        beta = 4  # from PiBO paper
        num_configs = len(self.observed_configs) + 1
        decay_rate = score ** (beta * (self.max_rung - rung) / num_configs)
        return decay_rate


class OurOptimizerV1_3(OurOptimizerV1):
    def calc_decay(self, score: float, rung: int) -> float:
        """Decays the prior score like PiBO but with a multi-fidelity decay.

        Pi(x) ^ (Beta * (z_max - z) / n_z); where n_z is the number of configurations seen
        in optimization history that has been evaluated at the fidelity z and z_max is
        the target fidelity. The exponent here is the same as PiBO but scaled by the
        difference in rung levels or discrete fidelity levels.

        Combines OptimizerV1 and OptimizerV1_2.
        OurOptimizerV1_3 = OptimizerV1 + OptimizerV1_2
        """
        beta = 4  # from PiBO paper
        rung_visits = self.find_rung_visits()  # contains the config count per fidelity
        num_configs = rung_visits[rung] + 1
        decay_rate = score ** (beta * (self.max_rung - rung) / num_configs)
        return decay_rate
