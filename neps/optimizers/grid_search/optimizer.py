import random
from typing import Any, override

from neps.state.optimizer import BudgetInfo
from neps.utils.types import ConfigResult, RawConfig
from neps.search_spaces.search_space import SearchSpace
from neps.optimizers.base_optimizer import BaseOptimizer


class GridSearch(BaseOptimizer):
    def __init__(
        self, pipeline_space: SearchSpace, grid_step_size: int = 10, **optimizer_kwargs
    ):
        super().__init__(pipeline_space=pipeline_space, **optimizer_kwargs)
        self._num_previous_configs: int = 0
        self.configs_list = self.pipeline_space.get_search_space_grid(
            size_per_numerical_hp=grid_step_size,
            include_endpoints=True,
        )
        random.shuffle(self.configs_list)

    @override
    def load_optimization_state(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        if self._num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")
        config = self.configs_list[self._num_previous_configs]
        config_id = str(self._num_previous_configs)
        return config.hp_values(), config_id, None
