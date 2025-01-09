from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scipy.sparse import random

from neps.optimizers.optimizer import SampledConfig

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


@dataclass
class GridSearch:
    pipeline_space: SearchSpace
    configs_list: list[dict[str, Any]]
    seed: int | None

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        _num_previous_configs = len(trials)
        if _num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")

        rng = random.Random(self.seed)
        configs = rng.sample(self.configs_list, len(self.configs_list))

        config = configs[_num_previous_configs]
        config_id = str(_num_previous_configs)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)
