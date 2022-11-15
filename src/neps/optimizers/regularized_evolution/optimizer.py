from __future__ import annotations

import random

from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer


class RegularizedEvolution(BaseOptimizer):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        population_size: int = 30,
        sample_size: int = 10,
        patience: int = 100,
        budget: None | int | float = None,
        logger=None,
        **optimizer_kwargs,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
            **optimizer_kwargs,
        )

        if population_size < 1:
            raise ValueError("RegularizedEvolution needs a population size >= 1")
        self.population_size = population_size
        self.sample_size = sample_size
        self.population: list = []

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        super().load_results(previous_results, pending_evaluations)
        train_x = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]
        self.population = list(
            zip(train_x[-self.population_size :], train_y[-self.population_size :])
        )

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if len(self.population) < self.population_size:
            config = self.random_sampler.sample(
                user_priors=True, constraint=self.sampling_constraint
            )
        else:
            candidates = [random.choice(self.population) for _ in range(self.sample_size)]
            parent = min(candidates, key=lambda c: c[1])[0]
            patience = self.patience
            while patience > 0:
                config = self._mutate(parent)
                if config is False:
                    config = self.random_sampler.sample(
                        user_priors=True,
                        constraint=self.sampling_constraint,
                    )
                if config not in self._pending_evaluations.values():
                    break
                patience -= 1
        return config, self.get_new_config_id(config), None

    def _mutate(self, parent):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent.mutate()
            except Exception:
                continue
        return False
