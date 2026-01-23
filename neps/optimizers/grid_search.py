from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import _get_max_trial_id

if TYPE_CHECKING:
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


@dataclass
class GridSearch:
    """Evaluates a fixed list of configurations in order."""

    configs_list: list[dict[str, Any]]
    """The list of configurations to evaluate."""

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        max_prev_trial_id = _get_max_trial_id(trials)
        if max_prev_trial_id > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")

        # TODO: Revisit this. Do we really need to shuffle the configs?
        configs = self.configs_list

        config = configs[max_prev_trial_id]
        config_id = str(max_prev_trial_id + 1)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)

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
