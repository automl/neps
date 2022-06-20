from __future__ import annotations

import logging
from typing import Any

import metahyper
from metahyper.api import ConfigResult

from ..search_spaces.search_space import SearchSpace
from ..utils.common import get_rnd_state, set_rnd_state
from ..utils.result_utils import get_cost, get_loss


class BaseOptimizer(metahyper.Sampler):
    """Base sampler class. Implements all the low-level work."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
        loss_value_on_error: float = float("inf"),
        cost_value_on_error: float = float("inf"),
    ):
        super().__init__(budget=budget)
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.pipeline_space = pipeline_space
        self.patience = patience
        self.logger = logger or logging.getLogger("neps")
        self.loss_value_on_error = loss_value_on_error
        self.cost_value_on_error = cost_value_on_error

        self._previous_results: dict[str, ConfigResult] = {}
        self._pending_evaluations: dict[str, SearchSpace] = {}

    @property
    def remaining_budget(self):
        return self.budget - self.used_budget

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        self._previous_results = previous_results
        self._pending_evaluations = pending_evaluations

    def get_new_config_id(self, config) -> str:  # pylint: disable=unused-argument
        """Used to generate the id of a new configuration when the returned id is None."""
        return str(len(self._previous_results) + len(self._pending_evaluations) + 1)

    def get_state(self) -> Any:  # pylint: disable=no-self-use
        return {
            "rnd_seeds": get_rnd_state(),
            **super().get_state(),
        }

    def load_state(self, state: Any):  # pylint: disable=no-self-use
        set_rnd_state(state["rnd_seeds"])
        super().load_state(state)

    def load_config(self, config_dict):
        config = self.pipeline_space.copy()
        config.load_from(config_dict)
        return config

    def get_loss(self, result: str | dict | float) -> float | Any:
        """Calls result.utils.get_loss() and passes the error handling through.
        Please use self.get_loss() instead of get_loss() in all optimizer classes."""
        return get_loss(result, loss_value_on_error=self.loss_value_on_error)

    def get_cost(self, result: str | dict | float) -> float | Any:
        """Calls result.utils.get_cost() and passes the error handling through.
        Please use self.get_cost() instead of get_cost() in all optimizer classes."""
        return get_cost(result, cost_value_on_error=self.cost_value_on_error)
