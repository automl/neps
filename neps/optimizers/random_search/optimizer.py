from __future__ import annotations

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self._num_previous_configs: int = 0

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config = self.pipeline_space.sample(patience=self.patience, use_user_priors=True)
        config_id = str(self._num_previous_configs + 1)
        return config, config_id, None
