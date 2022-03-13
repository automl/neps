from __future__ import annotations

import logging
import random
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Mapping

import metahyper
import torch
from metahyper.api import ConfigResult

from ..search_spaces.search_space import SearchSpace
from ..utils.common import get_rnd_state, set_rnd_state
from ..utils.result_utils import get_loss


class BaseOptimizer(metahyper.Sampler):
    """Base sampler class. Implements all the low-level work."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 0,
        random_interleave_prob: float = 0.0,
        patience: int = 50,
        logger=None,
        cost_function: None | Mapping = None,  # pylint: disable=unused-argument
        max_cost_total: None | int | float = None,  # pylint: disable=unused-argument
    ):
        if not 0 <= random_interleave_prob <= 1:
            raise ValueError("random_interleave_prob should be between 0.0 and 1.0")
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.pipeline_space = pipeline_space
        self.train_x: list = []
        self.train_y: list | torch.Tensor = []
        self.pending_evaluations: list = []

        self.initial_design_size = initial_design_size
        self.random_interleave_prob = random_interleave_prob
        self.patience = patience
        self.logger = logger or logging.getLogger("neps")

        self._model_update_failed = False

    @abstractmethod
    def sample(self) -> SearchSpace:
        raise NotImplementedError

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self.train_x = [el.config for el in previous_results.values()]
        self.train_y = [get_loss(el.result) for el in previous_results.values()]
        self.pending_evaluations = [el for el in pending_evaluations.values()]
        if len(self.train_x) >= self.initial_design_size:
            try:
                self._update_model()
                self._model_update_failed = False
            except RuntimeError:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model."
                )
                self._model_update_failed = True

    def _update_model(self):
        pass

    def get_state(self) -> Any:  # pylint: disable=no-self-use
        return {"rnd_seeds": get_rnd_state()}

    def load_state(self, state: Any):  # pylint: disable=no-self-use
        set_rnd_state(state["rnd_seeds"])

    def load_config(self, config_dict):
        config = deepcopy(self.pipeline_space)
        config.load_from(config_dict)
        return config

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if len(self.train_x) == 0 and self.initial_design_size >= 1:
            # TODO: if default config sample it
            config = self.pipeline_space.copy().sample(
                patience=self.patience, use_user_priors=True
            )
        elif random.random() < self.random_interleave_prob:
            config = self.pipeline_space.copy().sample(patience=self.patience)
        elif len(self.train_x) < self.initial_design_size or self._model_update_failed:
            config = self.pipeline_space.copy().sample(
                patience=self.patience, use_user_priors=True
            )
        else:
            for _ in range(self.patience):
                config = self.sample()
                if config not in self.pending_evaluations:
                    break
            else:
                config = self.pipeline_space.copy().sample(
                    patience=self.patience, use_user_priors=True
                )

        config_id = str(len(self.train_x) + len(self.pending_evaluations) + 1)
        return config, config_id, None
