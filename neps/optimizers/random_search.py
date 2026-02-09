from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
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
    constraints_func: Callable[[Mapping[str, Any]], Sequence[float]] | None = None
    raw_samples: int = 1024
    

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
        n_trials = len(trials)
        _n = 1 if n is None else n
        configs = self.sampler.sample(_n * (self.raw_samples or 1), to=self.encoder.domains)

        config_dicts = self.encoder.decode(configs)

        valid_configs = []
        for config in config_dicts:
            config.update(self.space.constants)
            if self.constraints_func is not None and self.constraints_func(config) < 0:
                continue
            if self.space.fidelity is not None:
                config.update(
                    {
                        key: value.upper
                        for key, value in self.space.fidelities.items()
                        if key not in config
                    }
                )
            valid_configs.append(config)
            if len(valid_configs) >= _n:
                break
        
        if self.constraints_func is not None and len(valid_configs) < _n:
            raise ValueError(
                f"Could not sample enough valid configurations out of "
                f"{_n * (self.raw_samples or 1)} samples, "
                f"only got {len(valid_configs)} valid ones."
            )
        
        valid_configs = valid_configs[:_n]

        if n is None:
            config = valid_configs[0]
            config_id = str(n_trials + 1)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(n_trials + i + 1),
                previous_config_id=None,
            )
            for i, config in enumerate(valid_configs)
        ]

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        n_trials = len(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(n_trials + i + 1)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
