"""NePS Algorithms
===========
This module provides implementations of various NePS algorithms for optimizing pipeline
spaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from neps.optimizers.neps_bracket_optimizer import _NePSBracketOptimizer
from neps.optimizers.neps_priorband import NePSPriorBandSampler
from neps.optimizers.neps_random_search import NePSComplexRandomSearch, NePSRandomSearch

if TYPE_CHECKING:
    import pandas as pd

    from neps.optimizers.utils.brackets import Bracket
    from neps.space.neps_spaces.parameters import Pipeline


def _neps_bracket_optimizer(
    pipeline_space: Pipeline,
    *,
    bracket_type: Literal["successive_halving", "hyperband", "asha", "async_hb"],
    eta: int,
    sampler: Literal["priorband"],
    sample_prior_first: bool | Literal["highest_fidelity"],
    early_stopping_rate: int | None,
) -> _NePSBracketOptimizer:
    fidelity_attrs = pipeline_space.fidelity_attrs

    if len(fidelity_attrs) != 1:
        raise ValueError(
            "Only one fidelity should be defined in the pipeline space."
            f"\nGot: {fidelity_attrs!r}"
        )

    fidelity_name, fidelity_obj = next(iter(fidelity_attrs.items()))

    if sample_prior_first not in (True, False, "highest_fidelity"):
        raise ValueError(
            "sample_prior_first should be either True, False or 'highest_fidelity'"
        )

    from neps.optimizers.utils import brackets

    # Determine the strategy for creating brackets for sampling
    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]
    match bracket_type:
        case "successive_halving":
            assert early_stopping_rate is not None
            rung_to_fidelity, rung_sizes = brackets.calculate_sh_rungs(
                bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Sync.create_repeating,
                rung_sizes=rung_sizes,
            )

        case "hyperband":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
                eta=eta,
            )
            create_brackets = partial(
                brackets.Hyperband.create_repeating,
                bracket_layouts=bracket_layouts,
            )

        case "asha":
            assert early_stopping_rate is not None
            rung_to_fidelity, _rung_sizes = brackets.calculate_sh_rungs(
                bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Async.create,
                rungs=list(rung_to_fidelity),
                eta=eta,
            )

        case "async_hb":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
                eta=eta,
            )
            # We don't care about the capacity of each bracket, we need the rung layout
            bracket_rungs = [list(bracket.keys()) for bracket in bracket_layouts]
            create_brackets = partial(
                brackets.AsyncHyperband.create,
                bracket_rungs=bracket_rungs,
                eta=eta,
            )
        case _:
            raise ValueError(f"Unknown bracket type: {bracket_type}")

    _sampler: NePSPriorBandSampler
    match sampler:
        case "priorband":
            _sampler = NePSPriorBandSampler(
                space=pipeline_space,
                eta=eta,
                early_stopping_rate=(
                    early_stopping_rate if early_stopping_rate is not None else 0
                ),
                fid_bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
            )
        case _:
            raise ValueError(f"Unknown sampler: {sampler}")

    return _NePSBracketOptimizer(
        space=pipeline_space,
        eta=eta,
        rung_to_fid=rung_to_fidelity,
        sampler=_sampler,
        sample_prior_first=sample_prior_first,
        create_brackets=create_brackets,
    )


def neps_priorband(
    space: Pipeline,
    *,
    eta: int = 3,
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
    base: Literal["successive_halving", "hyperband", "asha", "async_hb"] = "hyperband",
) -> _NePSBracketOptimizer:
    """Create a PriorBand optimizer for the given pipeline space.

    Args:
        space: The pipeline space to optimize over.
        eta: The eta parameter for the algorithm.
        sample_prior_first: Whether to sample the prior first.
            If set to `"highest_fidelity"`, the prior will be sampled at the
            highest fidelity, otherwise at the lowest fidelity.
        base: The type of bracket optimizer to use. One of:
            - "successive_halving"
            - "hyperband"
            - "asha"
            - "async_hb"
    Returns:
        An instance of _BracketOptimizer configured for PriorBand sampling.
    """
    return _neps_bracket_optimizer(
        pipeline_space=space,
        bracket_type=base,
        eta=eta,
        sampler="priorband",
        sample_prior_first=sample_prior_first,
        early_stopping_rate=0 if base in ("successive_halving", "asha") else None,
    )


def neps_random_search(
    pipeline: Pipeline,
    *_args: Any,
    **_kwargs: Any,
) -> NePSRandomSearch:
    """A simple random search algorithm that samples configurations uniformly at random.

    Args:
        pipeline: The search space to sample from.
    """

    return NePSRandomSearch(
        pipeline=pipeline,
    )


def neps_complex_random_search(
    pipeline: Pipeline,
    *_args: Any,
    **_kwargs: Any,
) -> NePSComplexRandomSearch:
    """A complex random search algorithm that samples configurations uniformly at random,
    but allows for more complex sampling strategies.

    Args:
        pipeline: The search space to sample from.
    """

    return NePSComplexRandomSearch(
        pipeline=pipeline,
    )
