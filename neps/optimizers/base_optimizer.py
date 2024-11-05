from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neps.state.trial import Report, Trial

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.utils.types import ERROR, ResultDict


def _get_loss(
    result: ERROR | ResultDict | float,
    loss_value_on_error: float | None = None,
    *,
    ignore_errors: bool = False,
) -> ERROR | float:
    if result == "error":
        if ignore_errors:
            return "error"

        if loss_value_on_error is not None:
            return loss_value_on_error

        raise ValueError(
            "An error happened during the execution of your evaluate_pipeline function."
            " You have three options: 1. If the error is expected and corresponds to"
            " a loss value in your application (e.g., 0% accuracy), you can set"
            " loss_value_on_error to some float. 2. If sometimes your pipeline"
            " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
        )

    if isinstance(result, dict):
        return float(result["loss"])

    assert isinstance(result, float)
    return float(result)


def _get_cost(
    result: ERROR | ResultDict | float,
    cost_value_on_error: float | None = None,
    *,
    ignore_errors: bool = False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"

        if cost_value_on_error is None:
            raise ValueError(
                "An error happened during the execution of your evaluate_pipeline"
                " function. You have three options: 1. If the error is expected and"
                " corresponds to a cost value in your application, you can set"
                " cost_value_on_error to some float. 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )

        return cost_value_on_error

    if isinstance(result, Mapping):
        return float(result["cost"])

    return float(result)


@dataclass
class SampledConfig:
    id: str
    config: Mapping[str, Any]
    previous_config_id: str | None = None


class BaseOptimizer:
    """Base sampler class. Implements all the low-level work."""

    # TODO: Remove a lot of these init params
    # Ideally we just make this a `Protocol`, i.e. an interface
    # and it has no functionality
    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        patience: int = 50,
        logger: logging.Logger | None = None,
        budget: int | float | None = None,
        loss_value_on_error: float | None = None,
        cost_value_on_error: float | None = None,
        learning_curve_on_error: float | list[float] | None = None,
        ignore_errors: bool = False,
    ) -> None:
        if patience < 1:
            raise ValueError("Patience should be at least 1")

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
    ) -> SampledConfig:
        """Sample a new configuration.

        Args:
            trials: All of the trials that are known about.
            budget_info: information about the budget

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
        # use `Trial` and `Report`, they already take care of this and save having to do
        # this `_get_loss` at every call. We can also then just use `None` instead of
        # the string `"error"`
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
        # use `Trial` and `Report`, they already take care of this and save having to do
        # this `_get_loss` at every call
        if isinstance(result, Report):
            return result.loss if result.loss is not None else "error"

        return _get_cost(
            result,
            cost_value_on_error=self.cost_value_on_error,
            ignore_errors=self.ignore_errors,
        )

    def whoami(self) -> str:
        return type(self).__name__
