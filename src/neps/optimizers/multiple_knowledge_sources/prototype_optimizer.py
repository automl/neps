from __future__ import annotations

import logging
from typing import Any

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ...utils.data_loading import read_tasks_and_dev_stages_from_disk
from .. import BaseOptimizer


# TODO: Test if anything breaks after the recent changes
class KnowledgeSampling(BaseOptimizer):
    def load_prev_dev_tasks(self):
        self.prev_task_dev_results = read_tasks_and_dev_stages_from_disk(
            self.paths_prev_task_and_dev
        )

    def __init__(
        self,
        paths_prev_task_and_dev: list[str],
        user_prior: dict,
        **optimizer_kwargs,
    ):
        super().__init__(**optimizer_kwargs)
        self.prev_task_dev_search_space = self.pipeline_space.copy()
        self._num_previous_configs: int = 0
        self.paths_prev_task_and_dev = paths_prev_task_and_dev
        self.prev_task_dev_results = None
        self.prior_search_spaces: dict[int, Any] = {}
        self.load_prev_dev_tasks()
        self.calculate_defaults()
        self.pipeline_space.set_hyperparameters_from_dict(
            user_prior, delete_previous_defaults=True, delete_previous_values=True
        )

    def calculate_defaults(self):
        configs = self.prev_task_dev_results[self.prev_task_dev_id[0]][
            self.prev_task_dev_id[1]
        ]
        hp_values = configs[0].config
        self.prev_task_dev_search_space.set_hyperparameters_from_dict(
            hp_values, delete_previous_defaults=True, delete_previous_values=True
        )

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config = None
        i = self._num_previous_configs
        if i == 0:
            # User prior
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
            logging.info("Sampling from user prior")
        elif i == 1:
            # Tasks / dev steps
            config = self.prev_task_dev_search_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
            logging.info("Sampling in mode tasks from previous tasks / dev stage")
        else:
            # Random search
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            logging.info("Sampling from random search")

        hp_values = config.hp_values()

        config_id = str(self._num_previous_configs + 1)
        logging.info("Config-ID: " + config_id)
        logging.info("Config:")
        logging.info(hp_values)
        return hp_values, config_id, None
