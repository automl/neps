from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autoscan.optimizers.optimizer import (
    ImportedConfig,
    SampledConfig,
    ScalingOptimizer,
)
from autoscan.optimizers.utils.util import _get_max_trial_id, get_trial_config_unique_key

if TYPE_CHECKING:
    from autoscan.state import BudgetInfo, Trial
    from autoscan.state.pipeline_eval import UserResultDict


@dataclass
class GridSearch(ScalingOptimizer):
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

        constraints_func = self.budget_constraint(self.constraints_func)
        evaluated = {
            get_trial_config_unique_key(t.config)
            for t in trials.values()
            if t.config is not None
        }

        # Next unevaluated config that satisfies the constraints
        config = next(
            (
                c
                for c in self.configs_list
                if get_trial_config_unique_key(c) not in evaluated
                and (constraints_func is None or constraints_func(c) >= 0)
            ),
            None,
        )
        if config is None:
            raise ValueError("Grid search exhausted or no valid configs found!")

        max_id = _get_max_trial_id(trials)
        next_id = 0 if max_id is None else max_id + 1
        return SampledConfig(config=config, id=str(next_id), previous_config_id=None)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        max_prev_trial_id = _get_max_trial_id(trials)
        first_id = 0 if max_prev_trial_id is None else max_prev_trial_id + 1
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(first_id + i)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
