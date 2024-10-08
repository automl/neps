from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Mapping

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo


class GridSearch(BaseOptimizer):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        grid_step_size: int = 10,
        seed: int | None = None,
    ):
        super().__init__(pipeline_space=pipeline_space)
        self.configs_list = self.pipeline_space.get_search_space_grid(
            size_per_numerical_hp=grid_step_size,
            include_endpoints=True,
        )
        self.seed = seed
        random.Random(seed).shuffle(self.configs_list)

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> SampledConfig | tuple[SampledConfig, dict[str, Any]]:
        _num_previous_configs = len(trials)
        if _num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")

        config = self.configs_list[_num_previous_configs]
        config_id = str(_num_previous_configs)
        return SampledConfig(
            config=config.hp_values(),
            id=config_id,
            previous_config_id=None,
        )
