from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Any

import metahyper
from metahyper.api import ConfigResult

from ..search_spaces.search_space import SearchSpace
from ..utils.common import get_rnd_state, set_rnd_state


class BaseOptimizer(metahyper.Sampler):
    """Base sampler class. Implements all the low-level work."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
    ):
        super().__init__(budget=budget)
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.pipeline_space = pipeline_space
        self.patience = patience
        self.logger = logger or logging.getLogger("neps")

    @abstractmethod
    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        raise NotImplementedError

    def get_state(self) -> Any:  # pylint: disable=no-self-use
        return {
            "rnd_seeds": get_rnd_state(),
            **super().get_state(),
        }

    def load_state(self, state: Any):  # pylint: disable=no-self-use
        set_rnd_state(state["rnd_seeds"])
        super().load_state(state)

    def load_config(self, config_dict):
        config = deepcopy(self.pipeline_space)
        config.load_from(config_dict)
        return config
