"""This module provides multi-fidelity optimizers for NePS spaces.
It implements a bracket-based optimization strategy that samples configurations
from a prior band, allowing for efficient exploration of the search space.
It supports different bracket types such as successive halving, hyperband, ASHA,
and async hyperband, and can sample configurations at different fidelity levels.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

import neps.optimizers.bracket_optimizer as standard_bracket_optimizer
from neps.optimizers.neps_local_and_incumbent import NePSLocalPriorIncumbentSampler
from neps.optimizers.neps_priorband import NePSPriorBandSampler
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.brackets import PromoteAction, SampleAction
from neps.optimizers.utils.util import (
    get_config_key_to_id_mapping,
    get_trial_config_unique_key,
)
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.sampling import (
    DomainSampler,
    OnlyPredefinedValuesSampler,
    PriorOrFallbackSampler,
    RandomSampler,
)

if TYPE_CHECKING:
    from neps.optimizers.utils.brackets import Bracket
    from neps.space.neps_spaces.parameters import PipelineSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.pipeline_eval import UserResultDict
    from neps.state.trial import Trial


logger = logging.getLogger(__name__)


@dataclass
class _NePSBracketOptimizer:
    """The pipeline space to optimize over."""

    space: PipelineSpace

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
    sampler: NePSPriorBandSampler | DomainSampler | NePSLocalPriorIncumbentSampler

    """The name of the fidelity in the space."""
    fid_name: str

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"

        # If we have no trials, we either go with the prior or just a sampled config
        if len(trials) == 0:
            match self.sample_prior_first:
                case "highest_fidelity":  # fid_max
                    config = self._sample_prior(fidelity_level="max")
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"1_rung_{rung}", config=config)
                case True:  # fid_min
                    config = self._sample_prior(fidelity_level="min")
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"1__rung_{rung}", config=config)
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
                    id=f"{config_id}_rung_{new_rung}",
                    config=config,
                    previous_config_id=f"{config_id}_rung_{new_rung - 1}",
                )

            # We need to sample for a new rung.
            case SampleAction(rung=rung):
                if isinstance(self.sampler, NePSPriorBandSampler):
                    config = self.sampler.sample_config(table, rung=rung)
                elif isinstance(self.sampler, NePSLocalPriorIncumbentSampler):
                    config = self.sampler.sample_config(table)
                elif isinstance(self.sampler, DomainSampler):
                    environment_values = {}
                    fidelity_attrs = self.space.fidelity_attrs
                    assert len(fidelity_attrs) == 1, "TODO: [lum]"
                    for fidelity_name, _fidelity_obj in fidelity_attrs.items():
                        environment_values[fidelity_name] = self.rung_to_fid[rung]
                    _, resolution_context = neps_space.resolve(
                        self.space,
                        domain_sampler=self.sampler,
                        environment_values=environment_values,
                    )
                    config = neps_space.NepsCompatConverter.to_neps_config(  # type: ignore[assignment]
                        resolution_context
                    )
                    config = dict(**config)
                config = self._convert_to_another_rung(config=config, rung=rung)
                return SampledConfig(
                    id=f"{nxt_id}_rung_{rung}",
                    config=config,
                )

            case _:
                raise RuntimeError(f"Unknown bracket action: {next_action}")

    def _sample_prior(
        self,
        fidelity_level: Literal["min"] | Literal["max"],
    ) -> dict[str, Any]:
        # TODO: [lum] have a CenterSampler as fallback, not Random
        _try_always_priors_sampler = PriorOrFallbackSampler(
            fallback_sampler=RandomSampler(predefined_samplings={}),
            always_use_prior=True,
        )

        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            if fidelity_level == "max":
                _environment_values[fidelity_name] = fidelity_obj.upper
            elif fidelity_level == "min":
                _environment_values[fidelity_name] = fidelity_obj.lower
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
            domain_sampler=OnlyPredefinedValuesSampler(
                predefined_samplings=data.predefined_samplings,
            ),
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        rung_to_fid = self.rung_to_fid

        # Use trials_to_table to get all used config IDs
        table = standard_bracket_optimizer.trials_to_table(trials)
        used_ids = set(table.index.get_level_values("id").tolist())

        imported_configs = []
        config_to_id = get_config_key_to_id_mapping(table=table, fid_name=self.fid_name)

        for config, result in external_evaluations:
            fid_value = config[self.fid_name]
            # create a unique key for the config without the fidelity
            config_key = get_trial_config_unique_key(
                config=config, fid_name=self.fid_name
            )
            # Assign id if not already assigned
            if config_key not in config_to_id:
                next_id = max(used_ids, default=0) + 1
                config_to_id[config_key] = next_id
                used_ids.add(next_id)
            else:
                existing_id = config_to_id[config_key]
                # check if the other config with same key has the same fidelity
                try:
                    existing_config = table.xs(existing_id, level="id")["config"].iloc[0]
                    if existing_config[self.fid_name] == config[self.fid_name]:
                        logger.warning(
                            f"Duplicate configuration with same fidelity found: {config}"
                        )
                    continue
                except KeyError:
                    pass

            config_id = config_to_id[config_key]

            # Find the rung corresponding to the fidelity value in config
            rung = max((r for r, f in rung_to_fid.items() if f <= fid_value))
            trial_id = f"{config_id}_rung_{rung}"
            imported_configs.append(
                ImportedConfig(
                    id=trial_id,
                    config=copy.deepcopy(config),
                    result=copy.deepcopy(result),
                )
            )
        return imported_configs
