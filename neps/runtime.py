"""TODO."""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import time
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from portalocker import portalocker
from pathlib import Path
from filelock import FileLock

from neps.env import (
    FS_SYNC_GRACE_BASE,
    FS_SYNC_GRACE_INC,
    LINUX_FILELOCK_FUNCTION,
    MAX_RETRIES_CREATE_LOAD_STATE,
    MAX_RETRIES_GET_NEXT_TRIAL,
    MAX_RETRIES_WORKER_CHECK_SHOULD_STOP,
)
from neps.exceptions import (
    NePSError,
    TrialAlreadyExistsError,
    WorkerFailedToGetPendingTrialsError,
    WorkerRaiseError,
)
from neps.state import (
    BudgetInfo,
    DefaultReportValues,
    EvaluatePipelineReturn,
    NePSState,
    OnErrorPossibilities,
    OptimizationState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
    evaluate_trial,
)
from neps.status.status import post_run_csv, _initiate_summary_csv, status
from neps.utils.common import gc_disabled

if TYPE_CHECKING:
    from neps.optimizers import OptimizerInfo
    from neps.optimizers.optimizer import AskFunction

logger = logging.getLogger(__name__)


def _default_worker_name() -> str:
    isoformat = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return f"{os.getpid()}-{isoformat}"


_DDP_ENV_VAR_NAME = "NEPS_DDP_TRIAL_ID"


def _is_ddp_and_not_rank_zero() -> bool:
    import torch.distributed as dist

    # Check for environment variables typically set by DDP
    ddp_env_vars = ["WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    rank_env_vars = ["RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"]

    # Check if PyTorch distributed is initialized
    if (dist.is_available() and dist.is_initialized()) or all(
        var in os.environ for var in ddp_env_vars
    ):
        for var in rank_env_vars:
            rank = os.environ.get(var)
            if rank is not None:
                return int(rank) != 0
    return False


def _set_ddp_env_var(trial_id: str) -> None:
    """Sets an environment variable with current trial_id in a DDP setup."""
    os.environ[_DDP_ENV_VAR_NAME] = trial_id


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
def get_workers_neps_state() -> NePSState:
    """Get the worker's NePS state."""
    if _WORKER_NEPS_STATE is None:
        raise RuntimeError(
            "The worker's NePS state has not been set! This should only be called"
            " from within a `evaluate_pipeline` context. If you are not running a"
            " pipeline and you did not call this function (`get_workers_neps_state`)"
            " yourself, this is a bug and should be reported to NePS."
        )
    return _WORKER_NEPS_STATE


def _set_workers_neps_state(state: NePSState) -> None:
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
    _set_ddp_env_var(trial.id)
    yield

    _CURRENTLY_RUNNING_TRIAL_IN_PROCESS = None


# NOTE: This class is quite stateful and has been split up quite a bit to make testing
# interleaving of workers easier. This comes at the cost of more fragmented code.
@dataclass
class DefaultWorker:
    """A default worker for the NePS system.

    This is the worker that is used by default in the neps.run() loop.
    """

    state: NePSState
    """The state of the NePS system."""

    settings: WorkerSettings
    """The settings for the worker."""

    evaluation_fn: Callable[..., EvaluatePipelineReturn]
    """The evaluation function to use for the worker."""

    optimizer: AskFunction
    """The optimizer that is in use by the worker."""

    worker_id: str
    """The id of the worker."""

    worker_cumulative_eval_count: int = 0
    """The number of evaluations done by this worker."""

    worker_cumulative_eval_cost: float = 0.0
    """The cost of the evaluations done by this worker."""

    worker_cumulative_evaluation_time_seconds: float = 0.0
    """The time spent evaluating configurations by this worker."""

    _GRACE: ClassVar = FS_SYNC_GRACE_BASE

    @classmethod
    def new(
        cls,
        *,
        state: NePSState,
        optimizer: AskFunction,
        settings: WorkerSettings,
        evaluation_fn: Callable[..., EvaluatePipelineReturn],
        worker_id: str | None = None,
    ) -> DefaultWorker:
        """Create a new worker."""
        return DefaultWorker(
            state=state,
            optimizer=optimizer,
            settings=settings,
            evaluation_fn=evaluation_fn,
            worker_id=worker_id if worker_id is not None else _default_worker_name(),
        )

    def _check_worker_local_settings(
        self,
        *,
        time_monotonic_start: float,
        error_from_this_worker: Exception | None,
    ) -> str | Literal[False]:
        # NOTE: Sorry this code is kind of ugly but it's pretty straightforward, just a
        # lot of conditional checking and making sure to check cheaper conditions first.

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

        return False

    def _check_shared_error_stopping_criterion(self) -> str | Literal[False]:
        # We check this global error stopping criterion as it's much
        # cheaper than sweeping the state from all trials.
        if self.settings.on_error in (
            OnErrorPossibilities.RAISE_ANY_ERROR,
            OnErrorPossibilities.STOP_ANY_ERROR,
        ):
            err = self.state.lock_and_get_errors().latest_err_as_raisable()
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

        return False

    def _check_global_stopping_criterion(
        self,
        trials: Mapping[str, Trial],
    ) -> str | Literal[False]:
        if self.settings.max_evaluations_total is not None:
            if self.settings.include_in_progress_evaluations_towards_maximum:
                if self.optimizer.space.fidelities:
                    count = sum(
                        trial.report.cost
                        for _, trial in trials.items()
                        if trial.report is not None and trial.report.cost is not None
                    )
                    for name, fidelity_param in self.optimizer.space.fidelities.items():
                        count = math.ceil(count / fidelity_param.upper)
                else:
                    count = sum(
                        1
                        for _, trial in trials.items()
                        if trial.metadata.state
                        not in (Trial.State.PENDING, Trial.State.SUBMITTED)
                    )
                    
            else:
                # This indicates they have completed.
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

    @property
    def _requires_global_stopping_criterion(self) -> bool:
        return (
            self.settings.max_evaluations_total is not None
            or self.settings.max_cost_total is not None
            or self.settings.max_evaluation_time_total_seconds is not None
        )

    def _get_next_trial(self) -> Trial | Literal["break"]:
        # If there are no global stopping criterion, we can no just return early.
        with self.state._optimizer_lock.lock(worker_id=self.worker_id):
            # NOTE: It's important to release the trial lock before sampling
            # as otherwise, any other service, such as reporting the result
            # of a trial. Hence we do not lock these together with the above.
            # OPTIM: We try to prevent garbage collection from happening in here to
            # minimize time spent holding on to the lock.
            with self.state._trial_lock.lock(worker_id=self.worker_id), gc_disabled():
                # Give the file-system some time to sync if we encountered out-of-order
                # issues with this worker.
                if self._GRACE > 0:
                    time.sleep(self._GRACE)

                trials = self.state._trial_repo.latest()

                if self._requires_global_stopping_criterion:
                    should_stop = self._check_global_stopping_criterion(trials)
                    if should_stop is not False:
                        logger.info(should_stop)
                        return "break"

                pending_trials = [
                    trial
                    for trial in trials.values()
                    if trial.metadata.state == Trial.State.PENDING
                ]

                if len(pending_trials) > 0:
                    earliest_pending = sorted(
                        pending_trials,
                        key=lambda t: t.metadata.time_sampled,
                    )[0]
                    earliest_pending.set_evaluating(
                        time_started=time.time(),
                        worker_id=self.worker_id,
                    )
                    self.state._trial_repo.update_trial(
                        earliest_pending, hints="metadata"
                    )
                    logger.info(
                        "Worker '%s' picked up pending trial: %s.",
                        self.worker_id,
                        earliest_pending.id,
                    )
                    return earliest_pending

            sampled_trials = self.state._sample_trial(
                optimizer=self.optimizer,
                worker_id=self.worker_id,
                trials=trials,
                n=self.settings.batch_size,
            )
            if isinstance(sampled_trials, Trial):
                this_workers_trial = sampled_trials
            else:
                this_workers_trial = sampled_trials[0]
                sampled_trials[1:]

            with self.state._trial_lock.lock(worker_id=self.worker_id), gc_disabled():
                this_workers_trial.set_evaluating(
                    time_started=time.time(),
                    worker_id=self.worker_id,
                )
                try:
                    self.state._trial_repo.store_new_trial(sampled_trials)
                    if isinstance(sampled_trials, Trial):
                        logger.info(
                            "Worker '%s' sampled new trial: %s.",
                            self.worker_id,
                            this_workers_trial.id,
                        )
                    else:
                        logger.info(
                            "Worker '%s' sampled new trials: %s.",
                            self.worker_id,
                            ",".join(trial.id for trial in sampled_trials),
                        )
                    return this_workers_trial
                except TrialAlreadyExistsError as e:
                    if e.trial_id in trials:
                        raise RuntimeError(
                            f"The new sampled trial was given an id of {e.trial_id}, yet"
                            " this exists in the loaded in trials given to the optimizer."
                            " This is a bug with the optimizers allocation of ids."
                        ) from e

                    _grace = DefaultWorker._GRACE
                    _inc = FS_SYNC_GRACE_INC
                    logger.warning(
                        "The new sampled trial was given an id of '%s', which is not"
                        " one that was loaded in by the optimizer. This is usually"
                        " an indication that the file-system you are running on"
                        " is not atmoic in synchoronizing file operations."
                        " We have attempted to stabalize this but milage may vary."
                        " We are incrementing a grace period for file-locks from"
                        " '%s's to '%s's. You can control the initial"
                        " grace with 'NEPS_FS_SYNC_GRACE_BASE' and the increment with"
                        " 'NEPS_FS_SYNC_GRACE_INC'.",
                        e.trial_id,
                        _grace,
                        _grace + _inc,
                    )
                    DefaultWorker._GRACE = _grace + FS_SYNC_GRACE_INC
                    raise e

    # Forgive me lord, for I have sinned, this function is atrocious but complicated
    # due to locking.
    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Run the worker.

        Will keep running until one of the criterion defined by the `WorkerSettings`
        is met.
        """
        _set_workers_neps_state(self.state)

        main_dir = Path(self.state.path)
        if self.settings.write_summary_to_disk:
            full_df_path, short_path, csv_locker = _initiate_summary_csv(main_dir)

            # Create empty CSV files
            with csv_locker.lock():
                full_df_path.parent.mkdir(parents=True, exist_ok=True)
                full_df_path.touch(exist_ok=True)
                short_path.touch(exist_ok=True)

            summary_dir = main_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            improvement_trace_path = summary_dir / "best_config_trajectory.txt"
            improvement_trace_path.touch(exist_ok=True)
            best_config_path = summary_dir / "best_config.txt"
            best_config_path.touch(exist_ok=True)
            _trace_lock = FileLock(".trace.lock")
            _trace_lock_path = Path(str(_trace_lock.lock_file))
            _trace_lock_path.touch(exist_ok=True)

            all_best_configs = []
            logger.info("Summary files of evaluations can be found in folder `Summary` in the main directory: %s", main_dir)

        _best_score_so_far = float("inf")

        optimizer_name = self.state._optimizer_info["name"]
        logger.info("Using optimizer: %s", optimizer_name)

        _time_monotonic_start = time.monotonic()
        _error_from_evaluation: Exception | None = None

        _repeated_fail_get_next_trial_count = 0
        n_repeated_failed_check_should_stop = 0
        while True:
            try:
                # First check local worker settings
                should_stop = self._check_worker_local_settings(
                    time_monotonic_start=_time_monotonic_start,
                    error_from_this_worker=_error_from_evaluation,
                )
                if should_stop is not False:
                    logger.info(should_stop)
                    break

                # Next check global errs having occured
                should_stop = self._check_shared_error_stopping_criterion()
                if should_stop is not False:
                    logger.info(should_stop)
                    break

            except WorkerRaiseError as e:
                # If we raise a specific error, we should stop the worker
                raise e
            except Exception as e:
                # An unknown exception, check our retry countk
                n_repeated_failed_check_should_stop += 1
                if (
                    n_repeated_failed_check_should_stop
                    >= MAX_RETRIES_WORKER_CHECK_SHOULD_STOP
                ):
                    raise WorkerRaiseError(
                        f"Worker {self.worker_id} failed to check if it should stop"
                        f" {MAX_RETRIES_WORKER_CHECK_SHOULD_STOP} times in a row. Bailing"
                    ) from e

                logger.error(
                    "Unexpected error from worker '%s' while checking if it should stop.",
                    self.worker_id,
                    exc_info=True,
                )
                time.sleep(1)  # Help stagger retries
                continue

            # From here, we now begin sampling or getting the next pending trial.
            # As the global stopping criterion requires us to check all trials, and
            # needs to be in locked in-step with sampling and is done inside
            # _get_next_trial
            try:
                trial_to_eval = self._get_next_trial()
                if trial_to_eval == "break":
                    break
                _repeated_fail_get_next_trial_count = 0
            except Exception as e:
                _repeated_fail_get_next_trial_count += 1
                if isinstance(e, portalocker.exceptions.LockException):
                    logger.debug(
                        "Worker '%s': Timeout while trying to get the next trial to"
                        " evaluate. If you are using a model based optimizer, such as"
                        " Bayesian Optimization, this can occur as the number of"
                        " configurations get large. There's not much to do here"
                        " and we will retry to obtain the lock.",
                        self.worker_id,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "Worker '%s': Error while trying to get the next trial to"
                        " evaluate.",
                        self.worker_id,
                        exc_info=True,
                    )
                    time.sleep(1)  # Help stagger retries
                # NOTE: This is to prevent any infinite loops if we can't get a trial
                if _repeated_fail_get_next_trial_count >= MAX_RETRIES_GET_NEXT_TRIAL:
                    raise WorkerFailedToGetPendingTrialsError(
                        f"Worker {self.worker_id} failed to get pending trials"
                        f" {MAX_RETRIES_GET_NEXT_TRIAL} times in"
                        " a row. Bailing!"
                    ) from e

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
                evaluated_trial.metadata.state,
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

            # We do not retry this, as if some other worker has
            # managed to manipulate this trial in the meantime,
            # then something has gone wrong
            with self.state._trial_lock.lock(worker_id=self.worker_id):
                self.state._report_trial_evaluation(
                    trial=evaluated_trial,
                    report=report,
                    worker_id=self.worker_id,
                )
                # This is mostly for `tblogger`
                for _key, callback in _TRIAL_END_CALLBACKS.items():
                    callback(trial_to_eval)

            if report.objective_to_minimize is not None and report.err is None:
                new_score = report.objective_to_minimize
                if new_score < _best_score_so_far:
                    _best_score_so_far = new_score
                    logger.info(
                        "Evaluated trial: %s with objective %s is the new best trial.",
                        evaluated_trial.id,
                        new_score,
                    )
                    
                    if self.settings.write_summary_to_disk:
                        # Store in memory for later file re-writing
                        all_best_configs.append({
                            "score": new_score,
                            "trial_id": evaluated_trial.id,
                            "config": evaluated_trial.config
                        })

                        # Build trace text and best config text
                        trace_text = "Best configs and their objectives across evaluations:\n" + "-" * 80 + "\n"
                        for best in all_best_configs:
                            trace_text += (
                                f"Objective to minimize: {best['score']}\n"
                                f"Config ID: {best['trial_id']}\n"
                                f"Config: {best['config']}\n"
                                + "-" * 80 + "\n"
                            )

                        best_config = all_best_configs[-1]  # Latest best
                        best_config_text = (
                            f"# Best config:"
                            f"\n\n    Config ID: {best_config['trial_id']}"
                            f"\n    Objective to minimize: {best_config['score']}"
                            f"\n    Config: {best_config['config']}"
                        )

                        # Write files from scratch
                        with _trace_lock:
                            with open(improvement_trace_path, mode='w') as f:
                                f.write(trace_text)

                            with open(best_config_path, mode='w') as f:
                                f.write(best_config_text)

                if self.settings.write_summary_to_disk:
                    full_df, short = status(main_dir)
                    with csv_locker.lock():
                        full_df.to_csv(full_df_path)
                        short.to_frame().to_csv(short_path)

            logger.debug("Config %s: %s", evaluated_trial.id, evaluated_trial.config)
            logger.debug("Loss %s: %s", evaluated_trial.id, report.objective_to_minimize)
            logger.debug("Cost %s: %s", evaluated_trial.id, report.objective_to_minimize)
            logger.debug(
                "Learning Curve %s: %s", evaluated_trial.id, report.learning_curve
            )


def _launch_ddp_runtime(
    *,
    evaluation_fn: Callable[..., EvaluatePipelineReturn],
    optimization_dir: Path,
    default_report_values: DefaultReportValues,
) -> None:
    neps_state = NePSState.create_or_load(path=optimization_dir, load_only=True)

    prev_trial: Trial | None = None

    # TODO: This could accidentally spin lock if the break is never hit.
    # This is quite dangerous as it could look like the worker is running but
    # it's not actually doing anything.
    while True:
        current_eval_trials = neps_state.lock_and_get_current_evaluating_trials()

        # If the worker id on previous trial is the same as the current one,
        # only then evaluate it.
        if len(current_eval_trials) == 0:
            continue

        current_trial: Trial | None = None
        if prev_trial is None:
            # In the beginning, we simply read the current trial from the env variable
            current_id = os.getenv(_DDP_ENV_VAR_NAME, "").strip()
            if current_id == "":
                raise RuntimeError(
                    "In a pytorch-lightning DDP setup, the environment variable"
                    f" '{_DDP_ENV_VAR_NAME}' was not set. This is probably a bug"
                    " in NePS and should be reported."
                )

            current_trial = neps_state.lock_and_get_trial_by_id(current_id)

        else:
            for trial in current_eval_trials:
                if (
                    trial.metadata.evaluating_worker_id
                    == prev_trial.metadata.evaluating_worker_id
                ) and (trial.id != prev_trial.id):
                    current_trial = trial
                    break

        if current_trial is not None:
            evaluate_trial(
                current_trial,
                evaluation_fn=evaluation_fn,
                default_report_values=default_report_values,
            )
            prev_trial = current_trial


# TODO: This should be done directly in `api.run` at some point to make it clearer at an
# entryy point how the worker is set up to run if someone reads the entry point code.
def _launch_runtime(  # noqa: PLR0913
    *,
    evaluation_fn: Callable[..., EvaluatePipelineReturn],
    optimizer: AskFunction,
    optimizer_info: OptimizerInfo,
    optimization_dir: Path,
    max_cost_total: float | None,
    ignore_errors: bool = False,
    objective_value_on_error: float | None,
    cost_value_on_error: float | None,
    continue_until_max_evaluation_completed: bool,
    overwrite_optimization_dir: bool,
    max_evaluations_total: int | None,
    max_evaluations_for_worker: int | None,
    sample_batch_size: int | None,
    write_summary_to_disk: bool = True,
) -> None:
    default_report_values = DefaultReportValues(
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
        cost_if_not_provided=None,  # TODO: User can't specify yet
        learning_curve_on_error=None,  # TODO: User can't specify yet
        learning_curve_if_not_provided="objective_to_minimize",  # report the
        # objective_to_minimize as single value LC
    )

    if _is_ddp_and_not_rank_zero():
        # Do not launch a new worker if we are in a DDP setup and not rank 0
        _launch_ddp_runtime(
            evaluation_fn=evaluation_fn,
            optimization_dir=optimization_dir,
            default_report_values=default_report_values,
        )
        return

    if overwrite_optimization_dir and optimization_dir.exists():
        logger.info(
            f"Overwriting optimization directory '{optimization_dir}' as"
            " `overwrite_optimization_dir=True`."
        )
        shutil.rmtree(optimization_dir)

    for _retry_count in range(MAX_RETRIES_CREATE_LOAD_STATE):
        try:
            neps_state = NePSState.create_or_load(
                path=optimization_dir,
                load_only=False,
                optimizer_info=optimizer_info,
                optimizer_state=OptimizationState(
                    seed_snapshot=SeedSnapshot.new_capture(),
                    budget=(
                        BudgetInfo(
                            max_cost_total=max_cost_total,
                            used_cost_budget=0,
                            max_evaluations=max_evaluations_total,
                            used_evaluations=0,
                        )
                    ),
                    shared_state=None,  # TODO: Unused for the time being...
                ),
            )
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
            logger.debug(
                "Error while trying to create or load the NePS state. Retrying...",
                exc_info=True,
            )
    else:
        raise RuntimeError(
            "Failed to create or load the NePS state after"
            f" {MAX_RETRIES_CREATE_LOAD_STATE} attempts. Bailing!"
            " Please enable debug logging to see the errors that occured."
        )

    settings = WorkerSettings(
        on_error=(
            OnErrorPossibilities.IGNORE
            if ignore_errors
            else OnErrorPossibilities.RAISE_ANY_ERROR
        ),
        batch_size=sample_batch_size,
        default_report_values=default_report_values,
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
        write_summary_to_disk=write_summary_to_disk
    )

    # HACK: Due to nfs file-systems, locking with the default `flock()` is not reliable.
    # Hence, we overwrite `portalockers` lock call to use `lockf()` instead.
    # This is commeneted in their source code that this is an option to use, however
    # it's not directly advertised as a parameter/env variable or otherwise.
    import portalocker.portalocker as portalocker_lock_module

    try:
        import fcntl

        if LINUX_FILELOCK_FUNCTION.lower() == "flock":
            setattr(portalocker_lock_module, "LOCKER", fcntl.flock)  # type: ignore[attr-defined]
        elif LINUX_FILELOCK_FUNCTION.lower() == "lockf":
            setattr(portalocker_lock_module, "LOCKER", fcntl.lockf)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"Unknown file-locking function '{LINUX_FILELOCK_FUNCTION}'."
                " Must be one of 'flock' or 'lockf'."
            )
    except ImportError:
        pass

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaluation_fn,
        settings=settings,
    )
    worker.run()
