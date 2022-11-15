from __future__ import annotations

from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(
        self, use_priors=False, ignore_fidelity=True, **optimizer_kwargs
    ):  # pylint: disable=unused-argument
        super().__init__(**optimizer_kwargs)
        self.use_priors = use_priors

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config = self.random_sampler.sample(
            user_priors=self.use_priors, constraint=self.sampling_constraint
        )
        return config, self.get_new_config_id(config), None
