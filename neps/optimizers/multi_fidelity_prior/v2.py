import typing

import numpy as np
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.hyperband import AsynchronousHyperband
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy, RandomUniformPolicy


class OurOptimizerV2(AsynchronousHyperband):
    """Implements a modified Asynchronous Hyperband.

    Adapts the sampling distribution from which ASHA brackets to run are sampled.
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
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = None,
    ):
        super().__init__(
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
        )

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

        # main change to the sampling distribution happens here
        # the probability mass is squashed for the brackets where the minimum rung has yet
        # to see any promoted configuration, with the mass being equally re-distributed
        # among the other eligible brackets
        bracket_probs = self._redistribute_probs(bracket_probs)

        bracket_next = np.random.choice(range(self.max_rung + 1), p=bracket_probs)
        return bracket_next

    def _redistribute_probs(self, bracket_probs):
        assert len(bracket_probs) == (self.max_rung + 1)
        mass_to_redistribute = 0.0
        num_inactive_brackets = 0
        # ignore the first rung or the first full SH bracket
        # at all times, the first bracket should be eligible for sampling
        for rung in reversed(range(1, self.max_rung + 1)):
            # if a rung has not been recorded in the observation history, that only
            # implies no config has been promoted to that rung, and thereby higher rungs
            # thus we squash the probability or chance of selection of such a bracket to
            # zero and redistribute that mass to other brackets which have seen promotions
            if rung not in self.observed_configs.rung.values:
                mass_to_redistribute += bracket_probs[rung]
                bracket_probs[rung] = 0.0
                num_inactive_brackets += 1

        if mass_to_redistribute:
            avg_mass_gain = mass_to_redistribute / (
                len(bracket_probs) - num_inactive_brackets
            )
            for i, prob in enumerate(bracket_probs):  # range(len(bracket_probs)):
                if prob == 0:
                    # if rung i has not seen promotions, rung i+1 shouldn't too
                    continue
                bracket_probs[i] += avg_mass_gain
        return bracket_probs


class OurOptimizerV2_2(OurOptimizerV2):
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


class OurOptimizerV2_3(OurOptimizerV2_2):
    def _redistribute_probs(self, bracket_probs):
        assert len(bracket_probs) == (self.max_rung + 1)
        mass_to_redistribute = 0.0
        num_inactive_brackets = 0
        # ignore the first rung or the first full SH bracket
        # at all times, the first bracket should be eligible for sampling
        for rung in reversed(range(1, self.max_rung + 1)):
            # if a rung has not been recorded in the observation history, that only
            # implies no config has been promoted to that rung, and thereby higher rungs
            # thus we squash the probability or chance of selection of such a bracket to
            # zero and redistribute that mass to other brackets which have seen promotions
            if rung not in self.observed_configs.rung.values:
                mass_to_redistribute += bracket_probs[rung]
                bracket_probs[rung] = 0.0
                num_inactive_brackets += 1
            elif len(np.where(self.observed_configs.rung.values == rung)[0]) < self.eta:
                # TODO: check this elif block
                #  Might become too similar to ASHA by forcing most samples at rung 0.
                #  However, this also ensures in some ways that a random sample is
                #  introduced at a rung only when the rung is wide enough (eta-width).
                #  But, for multiple workers, can afford to sample different brackets.
                mass_to_redistribute += bracket_probs[rung]
                bracket_probs[rung] = 0.0
                num_inactive_brackets += 1

        if mass_to_redistribute:
            avg_mass_gain = mass_to_redistribute / (
                len(bracket_probs) - num_inactive_brackets
            )
            for i, prob in enumerate(bracket_probs):  # range(len(bracket_probs)):
                if prob == 0:
                    # if rung i has not seen promotions, rung i+1 shouldn't too
                    continue
                bracket_probs[i] += avg_mass_gain
        return bracket_probs
