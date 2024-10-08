from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


class RandomSearch(BaseOptimizer):
    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        use_priors: bool = False,
        ignore_fidelity: bool = True,
        seed: int | None = None,
    ):
        super().__init__(pipeline_space=pipeline_space)
        self.use_priors = use_priors
        self.ignore_fidelity = ignore_fidelity
        if seed is not None:
            raise NotImplementedError("Seed is not implemented yet for RandomSearch")

        self.seed = seed

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> SampledConfig:
        # TODO: Replace with sampler objects and not `pipeline_space.sample`
        n_trials = len(trials)
        config = self.pipeline_space.sample(
            patience=self.patience,
            user_priors=self.use_priors,
            ignore_fidelity=self.ignore_fidelity,
        )
        config_id = str(n_trials + 1)
        return SampledConfig(
            config=config.hp_values(), id=config_id, previous_config_id=None
        )
