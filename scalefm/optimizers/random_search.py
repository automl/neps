from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scalefm.exceptions import ConstraintViolationError
from scalefm.optimizers.optimizer import (
    ImportedConfig,
    SampledConfig,
    ScalingOptimizer,
)
from scalefm.optimizers.utils.util import _get_next_trial_id
from scalefm.space.scalefm_spaces.parameters import PipelineSpace
from scalefm.space.scalefm_spaces.scalefm_space import (
    convert_scalefm_to_classic_search_space,
)

if TYPE_CHECKING:
    from scalefm.sampling import Sampler
    from scalefm.space import ConfigEncoder, SearchSpace
    from scalefm.state import BudgetInfo, Trial
    from scalefm.state.pipeline_eval import UserResultDict


@dataclass
class RandomSearch(ScalingOptimizer):
    """A simple random search optimizer."""

    space: SearchSpace | PipelineSpace
    encoder: ConfigEncoder
    sampler: Sampler
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None
    raw_samples: int = 1024
    constraint_cache: dict[str, bool] | None = None

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        if isinstance(self.space, PipelineSpace):
            converted_space = convert_scalefm_to_classic_search_space(self.space)
            if converted_space is not None:
                self.space = converted_space
            else:
                raise ValueError(
                    "This optimizer only supports HPO search spaces, please use an"
                    " ScaleFM"
                    " space-compatible optimizer."
                )
        next_id = _get_next_trial_id(trials)
        _n = 1 if n is None else n

        # Initialize constraint cache if needed
        if self.constraint_cache is None:
            self.constraint_cache = {}
        constraints_func = self.budget_constraint(self.constraints_func)

        # Generate fresh samples each time
        configs = self.sampler.sample(
            _n * (self.raw_samples or 1), to=self.encoder.domains
        )
        config_dicts = self.encoder.decode(configs)

        valid_configs = []
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

            # Check constraints on the exact config that may be returned.
            config_key = str(sorted(config.items()))
            if config_key not in self.constraint_cache:
                if constraints_func is not None:
                    result = constraints_func(config)
                    self.constraint_cache[config_key] = result >= 0
                else:
                    self.constraint_cache[config_key] = True

            if not self.constraint_cache[config_key]:
                continue

            valid_configs.append(config)
            if len(valid_configs) >= _n:
                break

        if constraints_func is not None and len(valid_configs) < _n:
            raise ConstraintViolationError(
                f"Could not sample enough valid configurations out of "
                f"{_n * (self.raw_samples or 1)} samples, "
                f"only got {len(valid_configs)} valid ones."
            )

        confs = valid_configs[:_n]

        if n is None or n == 1:
            config = confs[0]
            config_id = str(next_id)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(next_id + i),
                previous_config_id=None,
            )
            for i, config in enumerate(confs)
        ]

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        next_id = _get_next_trial_id(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(next_id + i)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
