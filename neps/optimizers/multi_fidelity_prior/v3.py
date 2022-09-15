import typing
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy, RandomUniformPolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalving


class OurOptimizerV3(AsynchronousSuccessiveHalving):
    """Implements a modified Asynchronous Hyperband.

    Alters the view of Async HB as a collection of ASHA brackets. Instead, samples the
    rung to query just as Mobster samples the ASHA bracket to run. For such a selected
    rung, a promotion happens if possible. Else a new sample is collected at that rung.
    """

    early_stopping_rate = 0

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RandomUniformPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
    ):
        super().__init__(
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
        )

    def _get_rung_to_run(self):
        """Samples the rung to run"""
        # Sampling distribution derived from Appendix A (https://arxiv.org/abs/2003.10865)
        # Adapting the distribution based on the current optimization state
        K = 5
        rung_probs = [
            self.eta ** (K - s) * (K + 1) / (K - s + 1) for s in range(self.max_rung + 1)
        ]
        rung_probs = np.array(rung_probs) / sum(rung_probs)
        bracket_next = np.random.choice(range(self.max_rung + 1), p=rung_probs)
        return bracket_next

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> Tuple[SearchSpace, str, Union[str, None]]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        if len(self.observed_configs.rung.unique()) < self.max_rung + 1:
            # call ASHA till every rung has seen at least one promoted configuration
            return super().get_config_and_ids()
        rung = self._get_rung_to_run()
        # becomes negative only when `rung` is 0
        # safe to assume 0 here as this class deals only with a full SH bracket
        rung_to_promote = max(0, rung - 1)

        if (
            rung_to_promote in self.rung_promotions
            and self.rung_promotions[rung_to_promote]
        ):
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            # if promotion check passes AND rung is 0 then a config from rung 0 is
            # being promoted to rung 1 --> handles rung = rung_to_promote
            rung = rung + 1 if rung == 0 else rung
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
        else:
            config = self.sample_new_config(rung=rung)
            previous_config_id = None
            config_id = f"{len(self.observed_configs)}_{rung}"

        # assigning the fidelity to evaluate the config at
        config.fidelity.value = self.rung_map[rung]

        return config.hp_values(), config_id, previous_config_id  # type: ignore


class OurOptimizerV3_2(OurOptimizerV3):
    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
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
        )
