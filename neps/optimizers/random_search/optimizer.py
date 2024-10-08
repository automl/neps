from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from neps.optimizers.base_optimizer import BaseOptimizer

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.utils.types import ConfigResult, RawConfig


class RandomSearch(BaseOptimizer):
    def __init__(self, use_priors=False, ignore_fidelity=True, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self._num_previous_configs: int = 0
        self.use_priors = use_priors
        self.ignore_fidelity = ignore_fidelity

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
        config = self.pipeline_space.sample(
            patience=self.patience,
            user_priors=self.use_priors,
            ignore_fidelity=self.ignore_fidelity,
        )
        config_id = str(self._num_previous_configs + 1)
        return config.hp_values(), config_id, None
