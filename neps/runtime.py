"""TODO."""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
)

from neps.exceptions import (
    NePSError,
    VersionMismatchError,
    WorkerFailedToGetPendingTrialsError,
    WorkerRaiseError,
)
from neps.state._eval import evaluate_trial
from neps.state.filebased import create_or_load_filebased_neps_state
from neps.state.optimizer import BudgetInfo, OptimizationState, OptimizerInfo
from neps.state.settings import DefaultReportValues, OnErrorPossibilities, WorkerSettings
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.optimizers.base_optimizer import BaseOptimizer
    from neps.state.neps_state import NePSState

logger = logging.getLogger(__name__)


def _default_worker_name() -> str:
    isoformat = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return f"{os.getpid()}-{isoformat}"


N_FAILED_GET_NEXT_PENDING_ATTEMPTS_BEFORE_ERROR = 0
N_FAILED_TO_SET_TRIAL_STATE = 10

Loc = TypeVar("Loc")

# NOTE: As each NEPS process is only ever evaluating a single trial, this global can
# be retrieved in NePS and refers to what this process is currently evaluating.
# Note that before `_set_in_progress_trial` is called, this should be cleared
# with `_clear_in_progress_trial` to ensure that we are not in some erroneuous state.
# Prefer to call `_clear_in_progress_trial` after a trial has finished evaluating and
# not just before `_set_in_progress_trial`, as the latter defeats the purpose of this
# assertion.
_CURRENTLY_RUNNING_TRIAL_IN_PROCESS: Trial | None = None
_WORKER_NEPS_STATE: NePSState | None = None


# TODO: This only works with a filebased nepsstate
def get_workers_neps_state() -> NePSState[Path]:
    """Get the worker's NePS state."""
    if _WORKER_NEPS_STATE is None:
        raise RuntimeError(
            "The worker's NePS state has not been set! This should only be called"
            " from within a `evaluate_pipeline` context. If you are not running a"
            " pipeline and you did not call this function (`get_workers_neps_state`)"
            " yourself, this is a bug and should be reported to NePS."
        )
    return _WORKER_NEPS_STATE


def _set_workers_neps_state(state: NePSState[Path]) -> None:
    global _WORKER_NEPS_STATE  # noqa: PLW0603
    _WORKER_NEPS_STATE = state


def get_in_progress_trial() -> Trial:
    """Get the currently running trial in this process."""
    if _CURRENTLY_RUNNING_TRIAL_IN_PROCESS is None:
        raise RuntimeError(
            "The worker's NePS state has not been set! This should only be called"
            " from within a `evaluate_pipeline` context. If you are not running a"
            " pipeline and you did not call this function (`get_workers_neps_state`)"
            " yourself, this is a bug and should be reported to NePS."
        )
    return _CURRENTLY_RUNNING_TRIAL_IN_PROCESS


_TRIAL_END_CALLBACKS: dict[str, Callable[[Trial], None]] = {}


def register_notify_trial_end(key: str, callback: Callable[[Trial], None]) -> None:
    """Register a callback to be called when a trial ends."""
    _TRIAL_END_CALLBACKS[key] = callback


@contextmanager
def _set_global_trial(trial: Trial) -> Iterator[None]:
    global _CURRENTLY_RUNNING_TRIAL_IN_PROCESS  # noqa: PLW0603
    if _CURRENTLY_RUNNING_TRIAL_IN_PROCESS is not None:
        raise NePSError(
            "A trial was already set to run in this process, yet some other trial was"
            " attempted to be set as the global trial in progress."
            " We assume that each process will only ever have one trial running at a time"
            " to allow functionality like `neps.get_in_progress_trial()`,"
            " `load_checkpoint()` and `save_checkpoint()` to work."
            "\n\nThis is most likely a bug and should be reported to NePS!"
        )
    _CURRENTLY_RUNNING_TRIAL_IN_PROCESS = trial
    yield
    for _key, callback in _TRIAL_END_CALLBACKS.items():
        callback(trial)
    _CURRENTLY_RUNNING_TRIAL_IN_PROCESS = None


# NOTE: This class is quite stateful and has been split up quite a bit to make testing
# interleaving of workers easier. This comes at the cost of more fragmented code.
@dataclass
class DefaultWorker(Generic[Loc]):
    """A default worker for the NePS system.

    This is the worker that is used by default in the neps.run() loop.
    """

    state: NePSState
    """The state of the NePS system."""

    settings: WorkerSettings
    """The settings for the worker."""

    evaluation_fn: Callable[..., float | Mapping[str, Any]]
    """The evaluation function to use for the worker."""

    optimizer: BaseOptimizer
    """The optimizer that is in use by the worker."""

    worker_id: str
    """The id of the worker."""

    _pre_sample_hooks: list[Callable[[BaseOptimizer], BaseOptimizer]] | None = None
    """Hooks to run before sampling a new trial."""

    worker_cumulative_eval_count: int = 0
    """The number of evaluations done by this worker."""

    worker_cumulative_eval_cost: float = 0.0
    """The cost of the evaluations done by this worker."""

    worker_cumulative_evaluation_time_seconds: float = 0.0
    """The time spent evaluating configurations by this worker."""

    @classmethod
    def new(
        cls,
        *,
        state: NePSState,
        optimizer: BaseOptimizer,
        settings: WorkerSettings,
        evaluation_fn: Callable[..., float | Mapping[str, Any]],
        _pre_sample_hooks: list[Callable[[BaseOptimizer], BaseOptimizer]] | None = None,
        worker_id: str | None = None,
    ) -> DefaultWorker:
        """Create a new worker."""
        return DefaultWorker(
            state=state,
            optimizer=optimizer,
            settings=settings,
            evaluation_fn=evaluation_fn,
            worker_id=worker_id if worker_id is not None else _default_worker_name(),
            _pre_sample_hooks=_pre_sample_hooks,
        )

    def _get_next_trial_from_state(self) -> Trial:
        nxt_trial = self.state.get_next_pending_trial()

        # If we have a trial, we will use it
        if nxt_trial is not None:
            logger.info(
                f"Worker '{self.worker_id}' got previosly sampled trial: {nxt_trial}"
            )

        # Otherwise sample a new one
        else:
            nxt_trial = self.state.sample_trial(
                worker_id=self.worker_id,
                optimizer=self.optimizer,
                _sample_hooks=self._pre_sample_hooks,
            )
            logger.info(f"Worker '{self.worker_id}' sampled a new trial: {nxt_trial}")

        return nxt_trial

    def _check_if_should_stop(  # noqa: C901, PLR0912, PLR0911
        self,
        *,
        time_monotonic_start: float,
        error_from_this_worker: Exception | None,
    ) -> str | Literal[False]:
        # NOTE: Sorry this code is kind of ugly but it's pretty straightforward, just a
        # lot of conditional checking and making sure to check cheaper conditions first.
        # It would look a little nicer with a match statement but we've got to wait
        # for python 3.10 for that.

        # First check for stopping criterion for this worker in particular as it's
        # cheaper and doesn't require anything from the state.
        if error_from_this_worker and self.settings.on_error in (
            OnErrorPossibilities.RAISE_WORKER_ERROR,
            OnErrorPossibilities.RAISE_ANY_ERROR,
            OnErrorPossibilities.STOP_WORKER_ERROR,
            OnErrorPossibilities.STOP_ANY_ERROR,
        ):
            msg = (
                "Error occurred while evaluating a configuration with this worker and"
                f" the worker is set to stop with {self.settings.on_error}."
                "\n"
                "\n"
                "If this was a bug in the evaluation code while you were developing your"
                " pipeline and you have set ignore_errors=True, please delete"
                " your results folder and fix the error before re-running."
                "\n"
                "If this is an issue specifically with the configuration, considering"
                " setting `ignore_errors=False` to allow the worker to continue"
                " evaluating other configurations, even if this one failed."
                "\n"
                "\n"
            )
            if self.settings.on_error in (
                OnErrorPossibilities.RAISE_WORKER_ERROR,
                OnErrorPossibilities.RAISE_ANY_ERROR,
            ):
                raise WorkerRaiseError(msg) from error_from_this_worker
            return msg

        if (
            self.settings.max_evaluations_for_worker is not None
            and self.worker_cumulative_eval_count
            >= self.settings.max_evaluations_for_worker
        ):
            return (
                "Worker has reached the maximum number of evaluations it is allowed to do"
                f" as given by `{self.settings.max_evaluations_for_worker=}`."
                "\nTo allow more evaluations, increase this value or use a different"
                " stopping criterion."
            )

        if (
            self.settings.max_cost_for_worker is not None
            and self.worker_cumulative_eval_cost >= self.settings.max_cost_for_worker
        ):
            return (
                "Worker has reached the maximum cost it is allowed to spend"
                f" which is given by `{self.settings.max_cost_for_worker=}`."
                f" This worker has spend '{self.worker_cumulative_eval_cost}'."
                "\n To allow more evaluations, increase this value or use a different"
                " stopping criterion."
            )

        if self.settings.max_wallclock_time_for_worker_seconds is not None and (
            time.monotonic() - time_monotonic_start
            >= self.settings.max_wallclock_time_for_worker_seconds
        ):
            return (
                "Worker has reached the maximum wallclock time it is allowed to spend"
                f", given by `{self.settings.max_wallclock_time_for_worker_seconds=}`."
            )

        if self.settings.max_evaluation_time_for_worker_seconds is not None and (
            self.worker_cumulative_evaluation_time_seconds
            >= self.settings.max_evaluation_time_for_worker_seconds
        ):
            return (
                "Worker has reached the maximum evaluation time it is allowed to spend"
                f", given by `{self.settings.max_evaluation_time_for_worker_seconds=}`."
            )

        # We check this global error stopping criterion as it's much
        # cheaper than sweeping the state from all trials.
        if self.settings.on_error in (
            OnErrorPossibilities.RAISE_ANY_ERROR,
            OnErrorPossibilities.STOP_ANY_ERROR,
        ):
            err = self.state._shared_errors.synced().latest_err_as_raisable()
            if err is not None:
                msg = (
                    "An error occurred in another worker and this worker is set to stop"
                    f" with {self.settings.on_error}."
                    "\n"
                    "If this was a bug in the evaluation code while you were developing"
                    " your pipeline and you have set ignore_errors=True, please delete"
                    " your results folder and fix the error before re-running."
                    "\n"
                    "If this is an issue specifically with the configuration, considering"
                    " setting `ignore_errors=False` to allow the worker to continue"
                    " evaluating other configurations, even if any worker fails."
                    "\n"
                )
                if self.settings.on_error == OnErrorPossibilities.RAISE_ANY_ERROR:
                    raise WorkerRaiseError(msg) from err

                return msg

        # If there are no global stopping criterion, we can no just return early.
        if (
            self.settings.max_evaluations_total is None
            and self.settings.max_cost_total is None
            and self.settings.max_evaluation_time_total_seconds is None
        ):
            return False

        # At this point, if we have some global stopping criterion, we need to sweep
        # the current state of trials to determine if we should stop
        # NOTE: If these `sum` turn out to somehow be a bottleneck, these could
        # be precomputed and accumulated over time. This would have to be handled
        # in the `NePSState` class.
        trials = self.state.get_all_trials()
        if self.settings.max_evaluations_total is not None:
            if self.settings.include_in_progress_evaluations_towards_maximum:
                count = sum(
                    1
                    for _, trial in trials.items()
                    if trial.report is not None
                    or trial.state in (Trial.State.EVALUATING, Trial.State.SUBMITTED)
                )
            else:
                count = sum(1 for _, trial in trials.items() if trial.report is not None)

            if count >= self.settings.max_evaluations_total:
                return (
                    "The total number of evaluations has reached the maximum allowed of"
                    f" `{self.settings.max_evaluations_total=}`."
                    " To allow more evaluations, increase this value or use a different"
                    " stopping criterion."
                )

        if self.settings.max_cost_total is not None:
            cost = sum(
                trial.report.cost
                for _, trial in trials.items()
                if trial.report is not None and trial.report.cost is not None
            )
            if cost >= self.settings.max_cost_total:
                return (
                    f"The maximum cost `{self.settings.max_cost_total=}` has been"
                    " reached by all of the evaluated trials. To allow more evaluations,"
                    " increase this value or use a different stopping criterion."
                )

        if self.settings.max_evaluation_time_total_seconds is not None:
            time_spent = sum(
                trial.report.evaluation_duration
                for _, trial in trials.items()
                if trial.report is not None
                if trial.report.evaluation_duration is not None
            )
            if time_spent >= self.settings.max_evaluation_time_total_seconds:
                return (
                    "The maximum evaluation time of"
                    f" `{self.settings.max_evaluation_time_total_seconds=}` has been"
                    " reached. To allow more evaluations, increase this value or use"
                    " a different stopping criterion."
                )

        return False

    def run(self) -> None:  # noqa: C901, PLR0915
        """Run the worker.

        Will keep running until one of the criterion defined by the `WorkerSettings`
        is met.
        """
        _set_workers_neps_state(self.state)

        logger.info("Launching NePS")

        _time_monotonic_start = time.monotonic()
        _error_from_evaluation: Exception | None = None

        _repeated_fail_get_next_trial_count = 0
        while True:
            # NOTE: We rely on this function to do logging and raising errors if it should
            should_stop = self._check_if_should_stop(
                time_monotonic_start=_time_monotonic_start,
                error_from_this_worker=_error_from_evaluation,
            )
            if should_stop is not False:
                logger.info(should_stop)
                break

            try:
                trial_to_eval = self._get_next_trial_from_state()
                _repeated_fail_get_next_trial_count = 0
            except Exception as e:
                _repeated_fail_get_next_trial_count += 1
                logger.debug(
                    "Error while trying to get the next trial to evaluate.", exc_info=True
                )

                # NOTE: This is to prevent any infinite loops if we can't get a trial
                if (
                    _repeated_fail_get_next_trial_count
                    >= N_FAILED_GET_NEXT_PENDING_ATTEMPTS_BEFORE_ERROR
                ):
                    raise WorkerFailedToGetPendingTrialsError(
                        "Worker '%s' failed to get pending trials %d times in a row."
                        " Bailing!"
                    ) from e

                continue

            # If we can't set this working to evaluating, then just retry the loop
            try:
                trial_to_eval.set_evaluating(
                    time_started=time.time(),
                    worker_id=self.worker_id,
                )
                self.state.put_updated_trial(trial_to_eval)
                n_failed_set_trial_state = 0
            except VersionMismatchError:
                n_failed_set_trial_state += 1
                logger.debug(
                    f"Another worker has managed to change trial '{trial_to_eval.id}'"
                    " to evaluate and put back into state. This is fine and likely means"
                    " the other worker is evaluating it.",
                    exc_info=True,
                )
            except Exception:
                n_failed_set_trial_state += 1
                logger.error(
                    f"Error trying to set trial '{trial_to_eval.id}' to evaluating.",
                    exc_info=True,
                )

            # NOTE: This is to prevent infinite looping if it somehow keeps getting
            # the same trial and can't set it to evaluating.
            if n_failed_set_trial_state != 0:
                if n_failed_set_trial_state >= N_FAILED_TO_SET_TRIAL_STATE:
                    raise WorkerFailedToGetPendingTrialsError(
                        "Worker '%s' failed to set trial to evaluating %d times in a row."
                        " Bailing!"
                    )
                continue

            # We (this worker) has managed to set it to evaluating, now we can evaluate it
            with _set_global_trial(trial_to_eval):
                evaluated_trial, report = evaluate_trial(
                    trial=trial_to_eval,
                    evaluation_fn=self.evaluation_fn,
                    default_report_values=self.settings.default_report_values,
                )
                evaluation_duration = evaluated_trial.metadata.evaluation_duration
                assert evaluation_duration is not None
                self.worker_cumulative_evaluation_time_seconds += evaluation_duration

            self.worker_cumulative_eval_count += 1

            logger.info(
                "Worker '%s' evaluated trial: %s as %s.",
                self.worker_id,
                evaluated_trial.id,
                evaluated_trial.state,
            )

            if report.cost is not None:
                self.worker_cumulative_eval_cost += report.cost

            if report.err is not None:
                logger.error(
                    f"Error during evaluation of '{evaluated_trial.id}'"
                    f" : {evaluated_trial.config}."
                )
                logger.exception(report.err)
                _error_from_evaluation = report.err

            self.state.report_trial_evaluation(
                trial=evaluated_trial,
                report=report,
                worker_id=self.worker_id,
            )

            logger.debug("Config %s: %s", evaluated_trial.id, evaluated_trial.config)
            logger.debug("Loss %s: %s", evaluated_trial.id, report.loss)
            logger.debug("Cost %s: %s", evaluated_trial.id, report.loss)
            logger.debug(
                "Learning Curve %s: %s", evaluated_trial.id, report.learning_curve
            )


# TODO: This should be done directly in `api.run` at some point to make it clearer at an
# entryy point how the woerer is set up to run if someone reads the entry point code.
def _launch_runtime(  # noqa: PLR0913
    *,
    evaluation_fn: Callable[..., float | Mapping[str, Any]],
    optimizer: BaseOptimizer,
    optimizer_info: dict,
    optimization_dir: Path,
    max_cost_total: float | None,
    ignore_errors: bool = False,
    loss_value_on_error: float | None,
    cost_value_on_error: float | None,
    continue_until_max_evaluation_completed: bool,
    overwrite_optimization_dir: bool,
    max_evaluations_total: int | None,
    max_evaluations_for_worker: int | None,
    pre_load_hooks: Iterable[Callable[[BaseOptimizer], BaseOptimizer]] | None,
) -> None:
    if overwrite_optimization_dir and optimization_dir.exists():
        logger.info(
            f"Overwriting optimization directory '{optimization_dir}' as"
            " `overwrite_optimization_dir=True`."
        )
        shutil.rmtree(optimization_dir)

    neps_state = create_or_load_filebased_neps_state(
        directory=optimization_dir,
        optimizer_info=OptimizerInfo(optimizer_info),
        optimizer_state=OptimizationState(
            budget=(
                BudgetInfo(
                    max_cost_budget=max_cost_total,
                    used_cost_budget=0,
                    max_evaluations=max_evaluations_total,
                    used_evaluations=0,
                )
            ),
            shared_state={},  # TODO: Unused for the time being...
        ),
    )

    settings = WorkerSettings(
        on_error=(
            OnErrorPossibilities.IGNORE
            if ignore_errors
            else OnErrorPossibilities.RAISE_ANY_ERROR
        ),
        default_report_values=DefaultReportValues(
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            cost_if_not_provided=None,  # TODO: User can't specify yet
            learning_curve_on_error=None,  # TODO: User can't specify yet
            learning_curve_if_not_provided="loss",  # report the loss as single value LC
        ),
        max_evaluations_total=max_evaluations_total,
        include_in_progress_evaluations_towards_maximum=(
            not continue_until_max_evaluation_completed
        ),
        max_cost_total=max_cost_total,
        max_evaluations_for_worker=max_evaluations_for_worker,
        max_evaluation_time_total_seconds=None,  # TODO: User can't specify yet
        max_wallclock_time_for_worker_seconds=None,  # TODO: User can't specify yet
        max_evaluation_time_for_worker_seconds=None,  # TODO: User can't specify yet
        max_cost_for_worker=None,  # TODO: User can't specify yet
    )

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaluation_fn,
        settings=settings,
        _pre_sample_hooks=list(pre_load_hooks) if pre_load_hooks is not None else None,
    )
    worker.run()
