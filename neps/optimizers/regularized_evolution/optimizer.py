import random
from collections import deque

from neps.optimizers.bayesian_optimization.acquisition_function_optimization.random_sampler import (
    RandomSampler,
)
from neps.optimizers.optimizer import Optimizer


class RegularizedEvolution(Optimizer):
    def __init__(
        self,
        pipeline_space,
        initial_population_size: int = 30,
        population_size: int = 30,
        sample_size: int = 10,
        patience: int = 50,
    ):
        super().__init__()
        self.initial_population_size = initial_population_size
        self.population_size = population_size
        self.sample_size = sample_size
        self.patience = patience
        self.search_space = pipeline_space
        self.random_sampler = RandomSampler(pipeline_space)

        self.population = deque()
        self.history = deque()
        self.tmp_counter = 0

    def get_config(self):
        if len(self.population) < self.initial_population_size:
            return self.random_sampler.sample(1)

        candidates = [random.choice(self.population) for _ in range(self.sample_size)]
        parent = min(candidates, key=lambda c: c["loss"])
        return self._mutate(parent["config"])

    def _propose_new_location(self, **kwargs):  # pylint: disable=W0613
        # only for compatability
        return [self.get_config()], None

    def _mutate(self, parent):
        _patience = self.patience
        while _patience > 0:
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                child = parent.mutate()
                return child
            except Exception:
                _patience -= 1
                continue
        return False

    def new_result(self, job):
        config = job["config"]
        loss = job["loss"]
        self._update(config, loss)

    def _initialize_model(self, x_configs, y):
        # only for compatability
        self.population = deque()
        self.history = deque()
        self.tmp_counter = 0
        self.update_model(x_configs, y)

    def _update_model(self, x_configs, y):
        # only for compatability
        while self.tmp_counter < len(x_configs):
            self.new_result(
                {"config": x_configs[self.tmp_counter], "loss": y[self.tmp_counter]}
            )
            self.tmp_counter += 1

    def _update(self, child, loss):
        self.history.append({"config": child, "loss": loss})
        self.population.append({"config": child, "loss": loss})
        if len(self.population) > self.population_size:
            self.population.popleft()

    def get_final_architecture(self):
        return max(self.history, key=lambda c: c[1])
