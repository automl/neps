from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neps.state.trial import Report, Trial
from neps.utils.data_loading import _get_cost, _get_learning_curve, _get_loss

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.utils.types import ERROR, ResultDict


@dataclass
class SampledConfig:
    id: str
    config: Mapping[str, Any]
    previous_config_id: str | None = None


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
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> SampledConfig | tuple[SampledConfig, dict[str, Any]]:
        """Sample a new configuration.

        !!! note

            The plan is this method replaces the two-step procedure of `load_optimization_state`
            and `get_config_and_ids` in the future, replacing both with a single method `ask`
            which would be easier for developer of NePS optimizers to implement.

        !!! note

            The `optimizer_state` right now is just a `dict` that optimizers are free to mutate
            as desired. A `dict` is not ideal as its _stringly_ typed but this was the least
            invasive way to add this at the moment. It's actually an existing feature no
            optimizer uses except _cost-cooling_ which basically just took a value from
            `budget_info`.

            Ideally an optimizer overwriting this can decide what to return instead of having
            to rely on them mutating it, however this is the best work-around I could come up with
            for now.

        Args:
            trials: All of the trials that are known about.
            budget_info: information about the budget
            optimizer_state: extra state the optimizer would like to keep between calls

        Returns:
            SampledConfig: a sampled configuration
            dict: state the optimizer would like to keep between calls
        """
        ...

    def get_loss(self, result: ERROR | ResultDict | float | Report) -> float | ERROR:
        """Calls result.utils.get_loss() and passes the error handling through.
        Please use self.get_loss() instead of get_loss() in all optimizer classes.
        """
        # TODO(eddiebergman): This is a forward change for whenever we can have optimizers
        # use `Trial` and `Report`, they already take care of this and save having to do this
        # `_get_loss` at every call. We can also then just use `None` instead of the string `"error"`
        if isinstance(result, Report):
            return result.loss if result.loss is not None else "error"

        return _get_loss(
            result,
            loss_value_on_error=self.loss_value_on_error,
            ignore_errors=self.ignore_errors,
        )

    def get_cost(self, result: ERROR | ResultDict | float | Report) -> float | ERROR:
        """Calls result.utils.get_cost() and passes the error handling through.
        Please use self.get_cost() instead of get_cost() in all optimizer classes.
        """
        # TODO(eddiebergman): This is a forward change for whenever we can have optimizers
        # use `Trial` and `Report`, they already take care of this and save having to do this
        # `_get_loss` at every call
        if isinstance(result, Report):
            return result.loss if result.loss is not None else "error"

        return _get_cost(
            result,
            cost_value_on_error=self.cost_value_on_error,
            ignore_errors=self.ignore_errors,
        )

    def get_learning_curve(
        self, result: str | dict | float | Report
    ) -> list[float] | Any:
        """Calls result.utils.get_loss() and passes the error handling through.
        Please use self.get_loss() instead of get_loss() in all optimizer classes.
        """
        # TODO(eddiebergman): This is a forward change for whenever we can have optimizers
        # use `Trial` and `Report`, they already take care of this and save having to do this
        # `_get_loss` at every call
        if isinstance(result, Report):
            return result.learning_curve

        return _get_learning_curve(
            result,
            learning_curve_on_error=self.learning_curve_on_error,
            ignore_errors=self.ignore_errors,
        )

    def whoami(self) -> str:
        return type(self).__name__
