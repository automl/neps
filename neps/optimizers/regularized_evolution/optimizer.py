""" Note: This optimizer is currently not supported.
"""

import random
from collections import deque
from typing import Any

import metahyper
import numpy as np
import torch
from deprecated import deprecated
from typing_extensions import Deque

from ..bayesian_optimization.acquisition_function_optimization.random_sampler import (
    RandomSampler,
)


class RegularizedEvolution(metahyper.Sampler):
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

        self.population: Deque = deque()
        self.history: Deque = deque()
        self.tmp_counter = 0

    def get_config_and_ids(self):
        raise NotImplementedError

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        raise NotImplementedError

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
        self._update_model(x_configs, y)

    @deprecated
    def _update_model(self, x_configs, y):
        # only for compatability
        while self.tmp_counter < len(x_configs):
            self.new_result(
                {"config": x_configs[self.tmp_counter], "loss": y[self.tmp_counter]}
            )
            self.tmp_counter += 1

    @deprecated
    def get_final_architecture(self):
        return max(self.history, key=lambda c: c[1])

    def get_state(self) -> Any:  # pylint: disable=no-self-use
        state = {
            "random_state": random.getstate(),
            "np_seed_state": np.random.get_state(),
            "torch_seed_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_seed_state"] = torch.cuda.get_rng_state_all()
        return state

    def load_state(self, state: Any):  # pylint: disable=no-self-use
        random.setstate(state["random_state"])
        np.random.set_state(state["np_seed_state"])
        torch.random.set_rng_state(state["torch_seed_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_seed_state"])
