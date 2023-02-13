from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

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
        assisted: bool = True,
        assisted_zero_cost_proxy: Callable | None = None,
        assisted_init_population_dir: str | Path | None = None,
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
        self.pending_evaluations: list = []
        self.num_train_x: int = 0

        self.assisted = assisted
        assert not assisted or (assisted and assisted_zero_cost_proxy is not None)
        self.assisted_zero_cost_proxy = assisted_zero_cost_proxy
        if assisted_init_population_dir is not None:
            self.assisted_init_population_dir = Path(assisted_init_population_dir)
            self.assisted_init_population_dir.mkdir(exist_ok=True)

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        train_x = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]
        self.num_train_x = len(train_x)
        self.population = [
            (x, y)
            for x, y in zip(
                train_x[-self.population_size :], train_y[-self.population_size :]
            )
        ]
        self.pending_evaluations = [el for el in pending_evaluations.values()]

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if len(self.population) < self.population_size:
            if self.assisted:
                if 0 == len(os.listdir(self.assisted_init_population_dir)):
                    print("Generate initial design with assistance")
                    cur_population_size = self.population_size - len(self.population)
                    configs = [
                        self.pipeline_space.sample(
                            patience=self.patience, user_priors=True
                        )
                        for _ in range(cur_population_size * 2)
                    ]
                    if self.assisted_zero_cost_proxy is not None:
                        zero_cost_proxy_values = self.assisted_zero_cost_proxy(
                            x=configs
                        )  # type:  ignore[misc]
                    else:
                        raise Exception("Zero cost proxy function is not defined!")
                    indices = np.argsort(zero_cost_proxy_values)[-cur_population_size:][
                        ::-1
                    ]
                    for idx, config_idx in enumerate(indices):
                        filename = str(idx).zfill(
                            int(math.log10(cur_population_size)) + 1
                        )
                        with open(
                            self.assisted_init_population_dir / f"{filename}.yaml",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            yaml.dump(configs[config_idx].serialize(), f)
                print("Pick config from pre-computed population")
                config_yaml = sorted(os.listdir(self.assisted_init_population_dir))[0]
                with open(
                    self.assisted_init_population_dir / config_yaml, encoding="utf-8"
                ) as f:
                    config_identifier = yaml.safe_load(f)
                config = self.pipeline_space.copy()
                config.load_from(config_identifier)
                os.remove(self.assisted_init_population_dir / config_yaml)
            else:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )
        else:
            candidates = [random.choice(self.population) for _ in range(self.sample_size)]
            parent = min(candidates, key=lambda c: c[1])[0]
            patience = self.patience
            while patience > 0:
                config = self._mutate(parent)
                if config is False:
                    config = self.pipeline_space.sample(
                        patience=self.patience, user_priors=True
                    )
                if config not in self.pending_evaluations:
                    break
                patience -= 1
        config_id = str(self.num_train_x + len(self.pending_evaluations) + 1)
        return config.hp_values(), config_id, None

    def _mutate(self, parent):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent.mutate()
            except Exception:
                continue
        return False
