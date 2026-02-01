from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import get_trial_config_unique_key
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
        _num_previous_configs = len(trials)
        
        # Find the next valid config respecting constraints
        config = None
        config_id = None
        for i in range(_num_previous_configs, len(self.configs_list)):
            candidate = self.configs_list[i]
            if self.constraints_func is not None and self.constraints_func(candidate) < 0:
                continue
            config = candidate
            config_id = str(i)
            break
        
        if config is None:
            raise ValueError("Grid search exhausted or no valid configs found!")

        return SampledConfig(config=config, id=config_id, previous_config_id=None)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        n_trials = len(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(n_trials + i)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
