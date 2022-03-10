""" Note: This optimizer is currently not supported.
"""

import random
from collections import deque

from deprecated import deprecated
from typing_extensions import Deque

from ..base_optimizer import BaseOptimizer
from ..bayesian_optimization.acquisition_samplers.random_sampler import RandomSampler


class RegularizedEvolution(BaseOptimizer):
    def __init__(
        self,
        initial_population_size: int = 30,
        population_size: int = 30,
        sample_size: int = 10,
        **optimizer_kwargs,
    ):
        super().__init__(**optimizer_kwargs)
        self.initial_population_size = initial_population_size
        self.population_size = population_size
        self.sample_size = sample_size
        self.random_sampler = RandomSampler()

        self.population: Deque = deque()
        self.history: Deque = deque()
        self.tmp_counter = 0
        self.random_sampler = RandomSampler()

    def _update_model(self):
        self.random_sampler.work_with(self.pipeline_space)

    def sample(self):
        raise NotImplementedError

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        raise NotImplementedError

    def _mutate(self, parent):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent.mutate()
            except Exception:
                continue
        return False

    def _update(self, child, loss):
        self.history.append({"config": child, "loss": loss})
        self.population.append({"config": child, "loss": loss})
        if len(self.population) > self.population_size:
            self.population.popleft()

    @deprecated
    def get_config(self):
        if len(self.population) < self.initial_population_size:
            return self.random_sampler.sample(1)[0]

        candidates = [random.choice(self.population) for _ in range(self.sample_size)]
        parent = min(candidates, key=lambda c: c["loss"])
        return self._mutate(parent["config"])

    @deprecated
    def _propose_new_location(self, **kwargs):  # pylint: disable=W0613
        # only for compatability
        return [self.get_config()], None

    @deprecated
    def new_result(self, job):
        config = job["config"]
        loss = job["loss"]
        self._update(config, loss)

    @deprecated
    def _initialize_model(self, x_configs, y):
        # only for compatability
        self.population = deque()
        self.history = deque()
        self.tmp_counter = 0
        self._update_model_old(x_configs, y)

    @deprecated
    def _update_model_old(self, x_configs, y):
        # only for compatability
        while self.tmp_counter < len(x_configs):
            self.new_result(
                {"config": x_configs[self.tmp_counter], "loss": y[self.tmp_counter]}
            )
            self.tmp_counter += 1

    @deprecated
    def get_final_architecture(self):
        return max(self.history, key=lambda c: c[1])
