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
    constraint_cache: dict[str, bool] | None = None
    

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
        
        # Initialize constraint cache if needed
        if self.constraint_cache is None:
            self.constraint_cache = {}
        
        # Generate fresh samples each time
        configs = self.sampler.sample(_n * (self.raw_samples or 1), to=self.encoder.domains)
        config_dicts = self.encoder.decode(configs)
        
        valid_configs = []
        for config in config_dicts:
            config.update(self.space.constants)
            
            # Check constraints using cache to avoid re-evaluation
            config_key = str(sorted(config.items()))
            if config_key not in self.constraint_cache:
                if self.constraints_func is not None:
                    result = self.constraints_func(config)
                    self.constraint_cache[config_key] = result >= 0
                else:
                    self.constraint_cache[config_key] = True
            
            if not self.constraint_cache[config_key]:
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
            print(f"RandomSearch: sampled config {config} with constraint value {len(valid_configs)} / {_n} valid so far.")
            if len(valid_configs) >= _n:
                break
        
        if self.constraints_func is not None and len(valid_configs) < _n:
            raise ValueError(
                f"Could not sample enough valid configurations out of "
                f"{_n * (self.raw_samples or 1)} samples, "
                f"only got {len(valid_configs)} valid ones."
            )

        confs = valid_configs[: _n]
        print(f"RandomSearch: returning {len(confs)} valid configurations out of {_n} requested.")
        

        if n is None or n == 1:
            config = confs[0]
            config_id = str(n_trials + 1)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(n_trials + i + 1),
                previous_config_id=None,
            )
            for i, config in enumerate(confs)
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
