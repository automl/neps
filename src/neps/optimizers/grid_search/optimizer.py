from __future__ import annotations

import random
from itertools import product

from ...search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
)
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class GridSearch(BaseOptimizer):
    def __init__(
        self, pipeline_space: SearchSpace, grid_step_size: int = 10, **optimizer_kwargs
    ):
        super().__init__(pipeline_space=pipeline_space, **optimizer_kwargs)
        self.grid_step_size = grid_step_size
        self.configs_list = self.make_search_grid()
        random.shuffle(self.configs_list)

    def make_search_grid(self):
        param_names, param_ranges = [], []
        for hp_name, hp in self.pipeline_space.items():
            param_names.append(hp_name)
            if isinstance(hp, CategoricalParameter):
                param_ranges.append(hp.choices)
            elif isinstance(hp, FloatParameter):
                if (
                    isinstance(hp, IntegerParameter)
                    and hp.upper - hp.lower < self.grid_step_size
                ):
                    grid = list(range(hp.lower, hp.upper + 1))
                else:
                    grid = [
                        hp.from_step(step, self.grid_step_size, in_place=False)
                        for step in range(self.grid_step_size)
                    ]
                param_ranges.append(grid)
            elif not isinstance(hp, ConstantParameter):
                raise ValueError(f"Can't use grid search on {hp.__class__}")
        full_grid = product(*param_ranges)

        configs = []
        for config_dict in full_grid:
            new_config = self.pipeline_space.copy()
            for hp_name, hp_value in zip(param_names, config_dict):
                new_config[hp_name].value = hp_value
            configs.append(new_config)
        return configs

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if not self.configs_list:
            raise ValueError("Grid search exhausted!")
        config = self.configs_list.pop()
        return config, self.get_new_config_id(config), None
