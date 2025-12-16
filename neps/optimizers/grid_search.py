from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

    non_evaluated_trials_key: list[tuple] = None

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        if self.none_evaluated_configs is None:
            self.none_evaluated_configs = [get_trial_config_unique_key(conf) for conf in self.configs_list]
            print("at first call, none evaluated configs:", len(self.none_evaluated_configs))
        _num_previous_configs = len(trials)
        if _num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")

        # TODO: Revisit this. Do we really need to shuffle the configs?
        configs = self.configs_list

        config = configs[_num_previous_configs]
        config_id = str(_num_previous_configs)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        n_trials = len(trials)
        imported_configs = []
        imported_keys = []
        for i, (config, result) in enumerate(external_evaluations):
            imported_keys.append(get_trial_config_unique_key(config=config))
            config_id = str(n_trials + i)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        self.none_evaluated_configs = [
            key for key in self.none_evaluated_configs if key not in imported_keys
        ]
        return imported_configs
