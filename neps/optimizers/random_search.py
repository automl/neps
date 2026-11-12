from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.initial_design import next_from_initial_design
from neps.optimizers.utils.util import _get_max_trial_id
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
from neps.space.neps_spaces.parameters import PipelineSpace

if TYPE_CHECKING:
    from neps.sampling import Sampler
    from neps.space import ConfigEncoder, SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


@dataclass
class RandomSearch:
    """A simple random search optimizer."""

    space: SearchSpace | PipelineSpace
    encoder: ConfigEncoder
    sampler: Sampler
    initial_design: list[dict[str, Any]] | None = None
    """Configs proposed first (completed with constants and fidelities), before
    sampling at random."""

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        if isinstance(self.space, PipelineSpace):
            converted_space = convert_neps_to_classic_search_space(self.space)
            if converted_space is not None:
                self.space = converted_space
            else:
                raise ValueError(
                    "This optimizer only supports HPO search spaces, please use a NePS"
                    " space-compatible optimizer."
                )
        max_trial_id = _get_max_trial_id(trials)
        _n = 1 if n is None else n
        design_configs = next_from_initial_design(self.initial_design, trials, _n)
        n_random = _n - len(design_configs)
        config_dicts = (
            self.encoder.decode(self.sampler.sample(n_random, to=self.encoder.domains))
            if n_random > 0
            else []
        )
        for config in config_dicts:
            config.update(self.space.constants)
            if self.space.fidelity is not None:
                config.update(
                    {
                        key: value.upper
                        for key, value in self.space.fidelities.items()
                        if key not in config
                    }
                )
        config_dicts = design_configs + config_dicts

        if n is None:
            config = config_dicts[0]
            config_id = str(max_trial_id + 1)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(max_trial_id + i + 1),
                previous_config_id=None,
            )
            for i, config in enumerate(config_dicts)
        ]

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        max_trial_id = _get_max_trial_id(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(max_trial_id + i + 1)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
