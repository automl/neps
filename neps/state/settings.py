"""Settings for the worker and the global state of NePS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass
class DefaultReportValues:
    """Values to use when an error occurs."""

    loss_value_on_error: float | None
    """The value to use for the loss when an error occurs."""

    cost_value_on_error: float | None
    """The value to use for the cost when an error occurs."""

    cost_if_not_provided: float | None
    """The value to use for the cost when the evaluation function does not provide one."""

    learning_curve_on_error: list[float] | None
    """The value to use for the learning curve when an error occurs.

    If `'loss'`, the learning curve will be set to the loss value but as
    a list with a single value.
    """

    learning_curve_if_not_provided: Literal["loss"] | list[float] | None = None
    """The value to use for the learning curve when the evaluation function does
    not provide one."""


class OnErrorPossibilities(Enum):
    """Possible values for what to do when an error occurs."""

    RAISE_WORKER_ERROR = "raise_worker_error"
    """Raise an error only if the error occurs in the worker."""

    STOP_WORKER_ERROR = "stop_worker_error"
    """Stop the worker if an error occurs in the worker, without raising"""

    RAISE_ANY_ERROR = "raise_any_error"
    """Raise an error if there was an error from any worker, i.e. there is a trial in the
    NePSState that has an error."""

    STOP_ANY_ERROR = "stop_any_error"
    """Stop the workers if any error occured from any worker, i.e. there is a trial in the
    NePSState that has an error."""

    IGNORE = "ignore"
    """Ignore all errors and continue running."""


# TODO: We can extend this over time
# For now this is what was needed for the backend state and workers.
@dataclass
class WorkerSettings:
    """Settings for a running instance of NePS."""

    # --------- Evaluation ---------
    on_error: OnErrorPossibilities
    """What to do when an error occurs.

    - `'raise_worker_error'`: Raise an error only if the error occurs in the worker.
    - `'raise_any_error'`: Raise an error if any error occurs from any worker, i.e.
        there is a trial in the NePSState that has an error.
    - `'ignore'`: Ignore all errors and continue running.
    """

    default_report_values: DefaultReportValues
    """Values to use when an error occurs or was not specified."""

    # --------- Global Stopping Criterion ---------
    max_evaluations_total: int | None
    """The maximum number of evaluations to run in total.

    Once this evaluation total is reached, **all** workers will stop evaluating
    new configurations.

    To control whether currently evaluating configurations are included in this
    total, see
    [`include_in_progress_evaluations_towards_maximum`][neps.state.settings.WorkerSettings.include_in_progress_evaluations_towards_maximum].

    If `None`, there is no limit and workers will continue to evaluate
    indefinitely.
    """

    include_in_progress_evaluations_towards_maximum: bool
    """Whether to include currently evaluating configurations towards the
    stopping criterion
    [`max_evaluations_total`][neps.state.settings.WorkerSettings.max_evaluations_total]
    """

    max_cost_total: float | None
    """The maximum cost to run in total.

    Once this cost total is reached, **all** workers will stop evaluating new
    configurations.

    This cost is the sum of `'cost'` values that are returned by evaluation
    of the target function.

    If `None`, there is no limit and workers will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """

    max_evaluation_time_total_seconds: float | None
    """The maximum wallclock time allowed for evaluation in total.

    !!! note
        This does not include time for sampling new configurations.

    Once this wallclock time is reached, **all** workers will stop once their
    current evaluation is finished.

    If `None`, there is no limit and workers will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """

    # --------- Local Worker Stopping Criterion ---------
    max_evaluations_for_worker: int | None
    """The maximum number of evaluations to run for the worker.

    This count is specific to each worker spawned by NePS.
    **only** the current worker will stop evaluating new configurations once
    this limit is reached.

    If `None`, there is no limit and this worker will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """

    max_cost_for_worker: float | None
    """The maximum cost incurred by a worker before finisihng.

    Once this cost total is reached, **only** this worker will stop evaluating new
    configurations.

    This cost is the sum of `'cost'` values that are returned by evaluation
    of the target function.

    If `None`, there is no limit and the worker will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """

    max_evaluation_time_for_worker_seconds: float | None
    """The maximum time to allow this worker for evaluating configurations.

    !!! note
        This does not include time for sampling new configurations.

    If `None`, there is no limit and this worker will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """

    max_wallclock_time_for_worker_seconds: float | None
    """The maximum wallclock time to run for this worker.

    Once this wallclock time is reached, **only** this worker will stop evaluating
    new configurations.

    !!! warning
        This will not stop the worker if it is currently evaluating a configuration.

    This is useful when the worker is deployed on some managed resource where
    there is a time limit.

    If `None`, there is no limit and this worker will continue to evaluate
    indefinitely or until another stopping criterion is met.
    """
