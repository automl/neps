from __future__ import annotations

import inspect
import logging
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from neps.exceptions import NePSError

if TYPE_CHECKING:
    from neps.state.settings import DefaultReportValues
    from neps.state.trial import Trial

logger = logging.getLogger(__name__)

Loc = TypeVar("Loc")
_notset = object()


class GotNonePendingTrialForEvalautionError(NePSError):
    """Raised when trying to evaluate a trial that is not in a pending state."""

    def __init__(
        self,
        trial_id: Trial.ID,
        state: Trial.State,
        worker_id: str,
        *args: Any,
    ):
        """Initialize the error.

        Args:
            trial_id: The ID of the trial that was not in a pending state.
            state: The state of the trial.
            worker_id: The ID of the worker that picked up this trial.
            *args: Additional arguments to pass to the parent class.
        """
        super().__init__(trial_id, state, worker_id, *args)
        self.trial_id = trial_id
        self.state = state
        self.worker_id = worker_id

    def __str__(self) -> str:
        return (
            f"Trial '{self.trial_id}' is not in a pending state but in '{self.state}'."
            f"This trial was picked up for evaluation by worker '{self.worker_id}'."
        )


def _check_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"The '{name}' should be a float but got a `{type(value)}`"
            f" with value of {value}",
        ) from e


def parse_user_result(
    user_result: float | dict[str, Any],
    *,
    default_cost_value: float | None = None,
    default_learning_curve: Literal["loss"] | list[float] | None = None,
) -> tuple[float, float | None, list[float] | None, dict[str, Any]]:
    """Check if the trial has succeeded."""
    if isinstance(user_result, Mapping):
        extracted_loss = user_result.pop("loss", _notset)
        if extracted_loss is _notset:
            raise KeyError(
                "The 'loss' should be provided in the evaluation result if providing"
                " a dictionary."
            )
        extracted_cost = user_result.pop("cost", default_cost_value)

        extracted_learning_curve = user_result.pop("learning_curve", _notset)

        if extracted_learning_curve is _notset:
            # HACK: Backwards compat, check if it's in the "info_dict" key
            if "info_dict" in user_result:
                extracted_learning_curve = user_result["info_dict"].pop(
                    "learning_curve",
                    default_learning_curve,
                )
            else:
                extracted_learning_curve = default_learning_curve

        if extracted_learning_curve == "loss":
            extracted_learning_curve = [extracted_loss]

        extra = user_result
    else:
        extracted_loss = user_result
        extracted_learning_curve = (
            None
            if default_learning_curve is None
            else [user_result]
            if default_learning_curve == "loss"
            else default_learning_curve
        )
        extracted_cost = default_cost_value
        extra = {}

    loss = _check_float(extracted_loss, "loss")
    cost = _check_float(extracted_cost, "cost") if extracted_cost is not None else None
    learning_curve = (
        [float(v) for v in extracted_learning_curve]
        if extracted_learning_curve is not None
        else None
    )
    return loss, cost, learning_curve, extra


def _eval_trial(
    *,
    trial: Trial,
    default_report_values: DefaultReportValues,
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Trial.Report:
    start = time.monotonic()
    try:
        user_result = fn(**kwargs, **trial.config)
    # Something went wrong in evaluation
    except Exception as e:
        duration = time.monotonic() - start
        time_end = time.time()
        logger.error(f"Error during evaluation of '{trial.id}': {trial.config}.")
        logger.exception(e)
        report = trial.set_complete(
            report_as="crashed",
            loss=default_report_values.loss_value_on_error,
            cost=default_report_values.cost_value_on_error,
            learning_curve=default_report_values.learning_curve_on_error,
            extra=None,
            err=e,
            tb=traceback.format_exc(),
            time_end=time_end,
            evaluation_duration=duration,
        )
    else:
        duration = time.monotonic() - start
        time_end = time.time()
        logger.info(f"Successful evaluation of '{trial.id}': {user_result}.")

        loss, cost, learning_curve, extra = parse_user_result(
            dict(user_result) if isinstance(user_result, Mapping) else user_result,
            default_cost_value=default_report_values.cost_if_not_provided,
            default_learning_curve=default_report_values.learning_curve_if_not_provided,
        )
        report = trial.set_complete(
            report_as="success",
            loss=loss,
            cost=cost,
            learning_curve=learning_curve,
            err=None,
            tb=None,
            extra=extra,
            time_end=time_end,
            evaluation_duration=duration,
        )

    return report


def evaluate_trial(
    trial: Trial,
    *,
    evaluation_fn: Callable[..., Any],
    default_report_values: DefaultReportValues,
) -> tuple[Trial, Trial.Report]:
    # NOTE: For now we are assuming everything is on a shared filesystem
    # will have to revisit if the location can be elsewhere
    trial_location = Path(trial.metadata.location)
    prev_trial_location = (
        Path(trial.metadata.previous_trial_location)
        if trial.metadata.previous_trial_location is not None
        else None
    )

    params = {
        "pipeline_directory": trial_location,
        "previous_pipeline_directory": prev_trial_location,
    }
    sigkeys = inspect.signature(evaluation_fn).parameters.keys()
    injectable_params = {key: val for key, val in params.items() if key in sigkeys}
    report = _eval_trial(
        trial=trial,
        fn=evaluation_fn,
        default_report_values=default_report_values,
        **injectable_params,
    )
    return trial, report
