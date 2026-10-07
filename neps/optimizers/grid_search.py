from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import get_trial_config_unique_key, _get_max_trial_id
if TYPE_CHECKING:
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


@dataclass
class GridSearch:
    """Evaluates a fixed list of configurations in order."""

    configs_list: list[dict[str, Any]]
    """The list of configurations to evaluate."""

    constraints_func: Callable[[Mapping[str, Any]], float] | None = field(default=None)
    """Optional constraint function that returns >= 0 for valid configs."""

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        
        # Build a mapping of config unique keys to trial IDs for quick lookup
        config_key_to_trial_id = {}
        for trial_id, trial in trials.items():
            if trial.config is not None:
                unique_key = get_trial_config_unique_key(trial.config)
                config_key_to_trial_id[unique_key] = trial_id
        
        # Get the maximum config_id assigned so far in our configs_list
        max_assigned_config_id = -1
        for trial_id, trial in trials.items():
            if trial.id is not None:
                try:
                    config_id = int(trial.id)
                    max_assigned_config_id = max(max_assigned_config_id, config_id)
                except (ValueError, TypeError):
                    pass
        
        next_config_id = max_assigned_config_id + 1
        
        # Find the next unvisited, valid config
        config = None
        for i, candidate in enumerate(self.configs_list):
            # Check if this config has already been evaluated
            unique_key = get_trial_config_unique_key(candidate)
            if unique_key in config_key_to_trial_id:
                continue  # Skip already evaluated config
            
            # Check if config passes constraints
            if self.constraints_func is not None and self.constraints_func(candidate) < 0:
                continue  # Skip configs that don't satisfy constraints
            
            config = candidate
            break
        
        if config is None:
            raise ValueError("Grid search exhausted or no valid configs found!")

        return SampledConfig(config=config, id=str(next_config_id), previous_config_id=None)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        max_prev_trial_id = _get_max_trial_id(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(max_prev_trial_id + i + 1)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
