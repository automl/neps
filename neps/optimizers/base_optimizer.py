from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Iterator, Mapping
from typing_extensions import Self
from contextlib import contextmanager
from pathlib import Path

from neps.types import ConfigResult
from neps.utils.files import serialize, deserialize
from ..search_spaces.search_space import SearchSpace
from ..utils.common import get_rnd_state, set_rnd_state
from neps.utils.data_loading import _get_cost, _get_learning_curve, _get_loss


class BaseOptimizer:
    """Base sampler class. Implements all the low-level work."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        patience: int = 50,
        logger: logging.Logger | None = None,
        budget: int | float | None = None,
        loss_value_on_error: float | None = None,
        cost_value_on_error: float | None = None,
        learning_curve_on_error: float | list[float] | None = None,
        ignore_errors=False,
    ) -> None:
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.used_budget: float = 0.0
        self.budget = budget
        self.pipeline_space = pipeline_space
        self.patience = patience
        self.logger = logger or logging.getLogger("neps")
        self.loss_value_on_error = loss_value_on_error
        self.cost_value_on_error = cost_value_on_error
        self.learning_curve_on_error = learning_curve_on_error
        self.ignore_errors = ignore_errors

    @abstractmethod
    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        """Sample a new configuration

        Returns:
            config: serializable object representing the configuration
            config_id: unique identifier for the configuration
            previous_config_id: if provided, id of a previous on which this
                configuration is based
        """
        raise NotImplementedError

    def get_state(self) -> Any:
        _state = {"rnd_seeds": get_rnd_state(), "used_budget": self.used_budget}
        if self.budget is not None:
            # TODO(eddiebergman): Seems like this isn't used anywhere,
            # A fuzzy find search for `remaining_budget` shows this as the
            # only use point.
            _state["remaining_budget"] = self.budget - self.used_budget

        return _state

    def load_state(self, state: Any) -> None:
        set_rnd_state(state["rnd_seeds"])
        self.used_budget = state["used_budget"]

    def load_config(self, config_dict: Mapping[str, Any]) -> SearchSpace:
        config = deepcopy(self.pipeline_space)
        config.load_from(config_dict)
        return config

    def get_loss(self, result: str | dict | float) -> float | Any:
        """Calls result.utils.get_loss() and passes the error handling through.
        Please use self.get_loss() instead of get_loss() in all optimizer classes."""
        return _get_loss(
            result,
            loss_value_on_error=self.loss_value_on_error,
            ignore_errors=self.ignore_errors,
        )

    def get_cost(self, result: str | dict | float) -> float | Any:
        """Calls result.utils.get_cost() and passes the error handling through.
        Please use self.get_cost() instead of get_cost() in all optimizer classes."""
        return _get_cost(
            result,
            cost_value_on_error=self.cost_value_on_error,
            ignore_errors=self.ignore_errors,
        )

    def get_learning_curve(self, result: str | dict | float) -> float | Any:
        """Calls result.utils.get_loss() and passes the error handling through.
        Please use self.get_loss() instead of get_loss() in all optimizer classes."""
        return _get_learning_curve(
            result,
            learning_curve_on_error=self.learning_curve_on_error,
            ignore_errors=self.ignore_errors,
        )

    def whoami(self) -> str:
        return type(self).__name__

    @contextmanager
    def using_state(self, state_file: Path) -> Iterator[Self]:
        if state_file.exists():
            state = deserialize(state_file)
            self.load_state(state)

        yield self

        serialize(self.get_state(), path=state_file)
