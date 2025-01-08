from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pandas.core.dtypes.missing import partial

from neps.optimizers.multi_fidelity._bracket_optimizer import BracketOptimizer
from neps.optimizers.multi_fidelity.brackets import (
    AsyncBracket,
    AsyncHyperbandBrackets,
    calculate_hb_bracket_layouts,
    calculate_sh_rungs,
)
from neps.search_spaces import SearchSpace

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace


logger = logging.getLogger(__name__)


class ASHA(BracketOptimizer):
    """Implements a ASHA algorithm."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        eta: int = 3,
        early_stopping_rate: int = 0,
        sampler: Literal["uniform", "prior", "priorband"] = "uniform",
        sample_default_first: bool | Literal["highest_fidelity"] = False,
        # TODO: Remove
        budget: int | float | None = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
    ):
        """Initialise an ASHA algorithm.

        Args:
            pipeline_space: Space in which to search
            eta: The reduction factor used by SH
            early_stopping_rate: Determines the number of rungs in an SH bracket
                Choosing 0 creates maximal rungs given the fidelity bounds
            sampler: The type of sampling procedure to use:

                * If "uniform", samples uniformly from the space when it needs to sample
                * If "prior", samples from the prior distribution built from the default
                  and default_confidence values in the pipeline space.
                * If "priorband", samples with weights according to the PriorBand
                    algorithm. See: https://arxiv.org/abs/2306.12370

            sample_default_first: Whether to sample the default configuration first.
        """
        assert pipeline_space.fidelity is not None
        rung_to_fidelity, rung_sizes = calculate_sh_rungs(
            bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
            eta=eta,
            early_stopping_rate=early_stopping_rate,
        )
        create_brackets = partial(
            AsyncBracket.make_asha_bracket,
            rungs=list(rung_to_fidelity),
            eta=eta,
        )
        super().__init__(
            pipeline_space=pipeline_space,
            eta=eta,
            sampler=sampler,
            sample_default_first=sample_default_first,
            create_brackets=create_brackets,
            rung_to_fidelity=rung_to_fidelity,
        )


class AsyncHB(BracketOptimizer):
    """Implements a Hyperband procedure."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        eta: int = 3,
        sampler: Literal["uniform", "prior", "priorband"] = "uniform",
        sample_default_first: bool | Literal["highest_fidelity"] = False,
        # TODO: Remove
        budget: int | float | None = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
    ):
        """Initialise an HyperBand algorithm.

        Args:
            pipeline_space: Space in which to search
            eta: The reduction factor used by HyperBand
            sampler: The type of sampling procedure to use:

                * If "uniform", samples uniformly from the space when it needs to sample
                * If "prior", samples from the prior distribution built from the default
                  and default_confidence values in the pipeline space.
                * If "priorband", samples with weights according to the PriorBand
                    algorithm. See: https://arxiv.org/abs/2306.12370

            sample_default_first: Whether to sample the default configuration first.
        """
        assert pipeline_space.fidelity is not None
        rung_to_fidelity, bracket_layouts = calculate_hb_bracket_layouts(
            bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
            eta=eta,
        )

        # We don't care about the capacity of each bracket, we need the rung layout
        bracket_rungs = [list(bracket.keys()) for bracket in bracket_layouts]

        super().__init__(
            pipeline_space=pipeline_space,
            create_brackets=partial(
                AsyncHyperbandBrackets.create,
                bracket_rungs=bracket_rungs,
                eta=eta,
            ),
            rung_to_fidelity=rung_to_fidelity,
            eta=eta,
            sampler=sampler,
            sample_default_first=sample_default_first,
        )
