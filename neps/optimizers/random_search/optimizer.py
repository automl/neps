from __future__ import annotations

from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        return config, self.get_new_config_id(config), None
