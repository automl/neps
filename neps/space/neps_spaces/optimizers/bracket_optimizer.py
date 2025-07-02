"""This module provides multi-fidelity optimizers for NePS spaces.
It implements a bracket-based optimization strategy that samples configurations
from a prior band, allowing for efficient exploration of the search space.
It supports different bracket types such as successive halving, hyperband, ASHA,
and async hyperband, and can sample configurations at different fidelity levels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

import neps.optimizers.bracket_optimizer as standard_bracket_optimizer
import neps.space.neps_spaces.sampling
from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.utils.brackets import PromoteAction, SampleAction
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.optimizers.priorband import PriorBandSampler

if TYPE_CHECKING:
    from neps.optimizers.utils.brackets import Bracket
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


logger = logging.getLogger(__name__)


@dataclass
class _BracketOptimizer:
    """The pipeline space to optimize over."""

    space: neps_space.Pipeline

    """Whether or not to sample the prior first.

    If set to `"highest_fidelity"`, the prior will be sampled at the highest fidelity,
    otherwise at the lowest fidelity.
    """
    sample_prior_first: bool | Literal["highest_fidelity"]

    """The eta parameter for the algorithm."""
    eta: int

    """The mapping from rung to fidelity value."""
    rung_to_fid: Mapping[int, int | float]

    """A function that creates the brackets from the table of trials."""
    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]

    """The sampler used to generate new trials."""
    sampler: PriorBandSampler

    def __call__(  # noqa: C901
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,  # noqa: ARG002
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"

        # If we have no trials, we either go with the prior or just a sampled config
        if len(trials) == 0:
            match self.sample_prior_first:
                case "highest_fidelity":  # fid_max
                    config = self._sample_prior(fidelity_level="max")
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"1_{rung}", config=config)
                case True:  # fid_min
                    config = self._sample_prior(fidelity_level="min")
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"1_{rung}", config=config)
                case False:
                    pass

        table = standard_bracket_optimizer.trials_to_table(trials=trials)

        if len(table) == 0:  # noqa: SIM108
            # Nothing there, this sample will be the first
            nxt_id = 1
        else:
            # One plus the maximum current id in the table index
            nxt_id = table.index.get_level_values("id").max() + 1  # type: ignore

        # We don't want the first highest fidelity sample ending
        # up in a bracket
        if self.sample_prior_first == "highest_fidelity":
            table = table.iloc[1:]

        # Get and execute the next action from our brackets that are not pending or done
        assert isinstance(table, pd.DataFrame)
        brackets = self.create_brackets(table)

        if not isinstance(brackets, Sequence):
            brackets = [brackets]

        next_action = next(
            (
                action
                for bracket in brackets
                if (action := bracket.next()) not in ("done", "pending")
            ),
            None,
        )

        if next_action is None:
            raise RuntimeError(
                f"{self.__class__.__name__} never got a 'sample' or 'promote' action!"
                f" This likely means the implementation of {self.create_brackets}"
                " is incorrect and should have provded enough brackets, where at"
                " least one of them should have requested another sample."
                f"\nBrackets:\n{brackets}"
            )

        match next_action:
            # The bracket would like us to promote a configuration
            case PromoteAction(config=config, id=config_id, new_rung=new_rung):
                config = self._convert_to_another_rung(config=config, rung=new_rung)
                return SampledConfig(
                    id=f"{config_id}_{new_rung}",
                    config=config,
                    previous_config_id=f"{config_id}_{new_rung - 1}",
                )

            # We need to sample for a new rung.
            case SampleAction(rung=rung):
                config = self.sampler.sample_config(table, rung=rung)
                config = self._convert_to_another_rung(config=config, rung=rung)
                return SampledConfig(
                    id=f"{nxt_id}_{rung}",
                    config=config,
                )

            case _:
                raise RuntimeError(f"Unknown bracket action: {next_action}")

    def _sample_prior(
        self,
        fidelity_level: Literal["min"] | Literal["max"],
    ) -> dict[str, Any]:
        # TODO: [lum] have a CenterSampler as fallback, not Random
        _try_always_priors_sampler = (
            neps.space.neps_spaces.sampling.PriorOrFallbackSampler(
                fallback_sampler=neps.space.neps_spaces.sampling.RandomSampler(
                    predefined_samplings={}
                ),
                prior_use_probability=1,
            )
        )

        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            if fidelity_level == "max":
                _environment_values[fidelity_name] = fidelity_obj.max_value
            elif fidelity_level == "min":
                _environment_values[fidelity_name] = fidelity_obj.min_value
            else:
                raise ValueError(f"Invalid fidelity level {fidelity_level}")

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=_try_always_priors_sampler,
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def _convert_to_another_rung(
        self,
        config: Mapping[str, Any],
        rung: int,
    ) -> dict[str, Any]:
        data = neps_space.NepsCompatConverter.from_neps_config(config=config)

        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        assert len(_fidelity_attrs) == 1, "TODO: [lum]"
        for fidelity_name, _fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = self.rung_to_fid[rung]

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
                predefined_samplings=data.predefined_samplings,
            ),
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)


def priorband(
    space: neps_space.Pipeline,
    *,
    eta: int = 3,
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
    base: Literal["successive_halving", "hyperband", "asha", "async_hb"] = "hyperband",
) -> _BracketOptimizer:
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
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type=base,
        eta=eta,
        sampler="priorband",
        sample_prior_first=sample_prior_first,
        early_stopping_rate=0 if base in ("successive_halving", "asha") else None,
    )


def _bracket_optimizer(
    pipeline_space: neps_space.Pipeline,
    *,
    bracket_type: Literal["successive_halving", "hyperband", "asha", "async_hb"],
    eta: int,
    sampler: Literal["priorband"],
    sample_prior_first: bool | Literal["highest_fidelity"],
    early_stopping_rate: int | None,
) -> _BracketOptimizer:
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

    _sampler: PriorBandSampler
    match sampler:
        case "priorband":
            _sampler = PriorBandSampler(
                space=pipeline_space,
                eta=eta,
                early_stopping_rate=(
                    early_stopping_rate if early_stopping_rate is not None else 0
                ),
                fid_bounds=(fidelity_obj.min_value, fidelity_obj.max_value),
            )
        case _:
            raise ValueError(f"Unknown sampler: {sampler}")

    return _BracketOptimizer(
        space=pipeline_space,
        eta=eta,
        rung_to_fid=rung_to_fidelity,
        sampler=_sampler,
        sample_prior_first=sample_prior_first,
        create_brackets=create_brackets,
    )
