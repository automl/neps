from __future__ import annotations

import random

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class GridSearch(BaseOptimizer):
    def __init__(
        self, pipeline_space: SearchSpace, grid_step_size: int = 10, **optimizer_kwargs
    ):
        super().__init__(pipeline_space=pipeline_space, **optimizer_kwargs)
        self._num_previous_configs: int = 0
        self.configs_list = self.pipeline_space.get_search_space_grid(
            grid_step_size=grid_step_size
        )
        random.shuffle(self.configs_list)

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if self._num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")
        config = self.configs_list[self._num_previous_configs]
        config_id = str(self._num_previous_configs)
        return config.hp_values(), config_id, None
