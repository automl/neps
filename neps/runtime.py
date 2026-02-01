"""Worker runtime implementation for NePS.

This module defines the default worker logic for running optimization trials in NePS.
It manages trial assignment, evaluation, stopping criteria, error handling, and
integration with distributed setups (e.g., PyTorch DDP).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from filelock import FileLock
from portalocker import portalocker

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
from neps.space.neps_spaces.neps_space import NepsCompatConverter, PipelineSpace
from neps.state import (
    BudgetInfo,
    DefaultReportValues,
    EvaluatePipelineReturn,
    NePSState,
    OnErrorPossibilities,
    OptimizationState,
    SeedSnapshot,
    Trial,
    UserResult,
    WorkerSettings,
    evaluate_trial,
)
from neps.utils.files import get_file_writer
from neps.status.status import (
    _build_incumbent_content,
    _build_optimal_set_content,
    _initiate_summary_csv,
    status,
)
from neps.utils.common import gc_disabled

if TYPE_CHECKING:
    from neps import SearchSpace
    from neps.optimizers import OptimizerInfo
    from neps.optimizers.optimizer import AskFunction

logger = logging.getLogger(__name__)


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


def is_in_progress_trial_set() -> bool:
    """Check if the currently running trial in this process is set."""
    return _CURRENTLY_RUNNING_TRIAL_IN_PROCESS is not None


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


@dataclass
class ResourceUsage:
    """Container for tracking cumulative resource usage."""

    evaluations: int = 0
    cost: float = 0.0
    fidelities: float = 0.0
    time: float = 0.0

    def __iadd__(self, other: ResourceUsage) -> ResourceUsage:
        """Allows syntax: usage += other_usage."""
        self.evaluations += other.evaluations
        self.cost += other.cost
        self.fidelities += other.fidelities
        self.time += other.time
        return self

    def to_trajectory_dict(self) -> dict[str, float | int]:
        """Converts usage to the dictionary keys expected by the trajectory file."""
        return {
            "cumulative_evaluations": self.evaluations,
            "cumulative_cost": self.cost,
            "cumulative_fidelities": self.fidelities,
            "cumulative_time": self.time,
        }


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
        worker_id = state.lock_and_set_new_worker_id(worker_id)
        return DefaultWorker(
            state=state,
            optimizer=optimizer,
            settings=settings,
            evaluation_fn=evaluation_fn,
            worker_id=worker_id,
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

        if self.settings.max_wallclock_time_seconds is not None and (
            time.monotonic() - time_monotonic_start
            >= self.settings.max_wallclock_time_seconds
        ):
            return (
                "Worker has reached the maximum wallclock time it is allowed to spend"
                f", given by `{self.settings.max_wallclock_time_seconds=}`."
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

    def _calculate_total_resource_usage(  # noqa: C901
        self,
        trials: Mapping[str, Trial],
        subset_worker_id: str | None = None,
        *,
        include_in_progress: bool = False,
    ) -> ResourceUsage:
        """Calculates total resources returning a typed usage object.

        Args:
            trials: Dictionary of trials to calculate from.
            subset_worker_id: If provided, only calculates for
                trials evaluated by this worker ID.
            include_in_progress: Whether to include incomplete trials.
        """
        relevant_trials = list(trials.values())
        if subset_worker_id is not None:
            relevant_trials = [
                t
                for t in relevant_trials
                if t.metadata.evaluating_worker_id == subset_worker_id
            ]

        fidelity_name = None
        if hasattr(self.optimizer, "space"):
            if isinstance(self.optimizer.space, PipelineSpace):
                if self.optimizer.space.fidelity_attrs:
                    fidelity_name = next(iter(self.optimizer.space.fidelity_attrs.keys()))
                    fidelity_name = (
                        f"{NepsCompatConverter._ENVIRONMENT_PREFIX}{fidelity_name}"
                    )
            elif self.optimizer.space.fidelities:
                fidelity_name = next(iter(self.optimizer.space.fidelities.keys()))

        usage = ResourceUsage()

        for trial in relevant_trials:
            if not (
                trial.report is not None
                or (
                    include_in_progress and trial.metadata.state == Trial.State.EVALUATING
                )
            ):
                continue
            usage.evaluations += 1
            if trial.report and trial.report.cost is not None:
                usage.cost += trial.report.cost

            # Handle time: either from report or calculate from metadata
            if trial.report and trial.report.evaluation_duration is not None:
                usage.time += trial.report.evaluation_duration
            elif (
                trial.metadata.time_started is not None
                and trial.metadata.time_end is not None
            ):
                usage.time += trial.metadata.time_end - trial.metadata.time_started

            if (
                fidelity_name
                and fidelity_name in trial.config
                and trial.config[fidelity_name] is not None
            ):
                usage.fidelities += trial.config[fidelity_name]

        return usage

    def _check_global_stopping_criterion(  # noqa: C901
        self,
        trials: Mapping[str, Trial],
        log_status: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[str | Literal[False], ResourceUsage]:
        """Evaluates if any global stopping criterion has been met.

        Args:
            trials: The trials to evaluate the stopping criterion on.
            log_status: Whether to log the current status of the budget.

        Returns:
            A tuple of (stopping message or False, global resource usage).
        """
        worker_resource_usage = self._calculate_total_resource_usage(
            trials,
            subset_worker_id=self.worker_id,
            include_in_progress=self.settings.include_in_progress_evaluations_towards_maximum,
        )

        global_resource_usage = self._calculate_total_resource_usage(
            trials,
            subset_worker_id=None,
            include_in_progress=self.settings.include_in_progress_evaluations_towards_maximum,
        )

        if log_status:
            # Log current budget status
            budget_info_parts = []
            if self.settings.evaluations_to_spend is not None:
                eval_percentage = int(
                    (
                        worker_resource_usage.evaluations
                        / self.settings.evaluations_to_spend
                    )
                    * 100
                )
                budget_info_parts.append(
                    "Evaluations:"
                    f" {worker_resource_usage.evaluations}/"
                    f"{self.settings.evaluations_to_spend}"
                    f" ({eval_percentage}%)"
                )
            if self.settings.fidelities_to_spend is not None:
                fidelity_percentage = int(
                    (worker_resource_usage.fidelities / self.settings.fidelities_to_spend)
                    * 100
                )
                budget_info_parts.append(
                    "Fidelities:"
                    f" {worker_resource_usage.fidelities}/"
                    f"{self.settings.fidelities_to_spend}"
                    f" ({fidelity_percentage}%)"
                )
            if self.settings.cost_to_spend is not None:
                cost_percentage = int(
                    (worker_resource_usage.cost / self.settings.cost_to_spend) * 100
                )
                budget_info_parts.append(
                    "Cost:"
                    f" {worker_resource_usage.cost}/"
                    f"{self.settings.cost_to_spend} ({cost_percentage}%)"
                )
            if self.settings.max_evaluation_time_total_seconds is not None:
                time_percentage = int(
                    (
                        worker_resource_usage.time
                        / self.settings.max_evaluation_time_total_seconds
                    )
                    * 100
                )
                budget_info_parts.append(
                    "Time:"
                    f" {worker_resource_usage.time}/"
                    f"{self.settings.max_evaluation_time_total_seconds}s"
                    f" ({time_percentage}%)"
                )

            if budget_info_parts:
                logger.info("Budget status - %s", " | ".join(budget_info_parts))
        return_string: str | Literal[False] = False

        if (
            self.settings.evaluations_to_spend is not None
            and worker_resource_usage.evaluations >= self.settings.evaluations_to_spend
        ):
            return_string = (
                "Worker has reached the maximum number of evaluations it is allowed"
                f" to do as given by `{self.settings.evaluations_to_spend=}`."
                "\nTo allow more evaluations, increase this value or use a different"
                " stopping criterion."
            )

        if (
            self.settings.fidelities_to_spend is not None
            and worker_resource_usage.fidelities >= self.settings.fidelities_to_spend
        ):
            return_string = (
                "The total number of fidelity evaluations has reached the maximum"
                f" allowed of `{self.settings.fidelities_to_spend=}`."
                " To allow more evaluations, increase this value or use a different"
                " stopping criterion."
            )

        if (
            self.settings.cost_to_spend is not None
            and worker_resource_usage.cost >= self.settings.cost_to_spend
        ):
            return_string = (
                "Worker has reached the maximum cost it is allowed to spend"
                f" which is given by `{self.settings.cost_to_spend=}`."
                f" This worker has spend '{worker_resource_usage.cost}'."
                "\n To allow more evaluations, increase this value or use a different"
                " stopping criterion."
            )

        if (
            self.settings.max_evaluation_time_total_seconds is not None
            and worker_resource_usage.time
            >= self.settings.max_evaluation_time_total_seconds
        ):
            return_string = (
                "The maximum evaluation time of"
                f" `{self.settings.max_evaluation_time_total_seconds=}` has been"
                " reached. To allow more evaluations, increase this value or use"
                " a different stopping criterion."
            )

        return (return_string, global_resource_usage)

    @property
    def _requires_global_stopping_criterion(self) -> bool:
        return (
            self.settings.evaluations_to_spend is not None
            or self.settings.cost_to_spend is not None
            or self.settings.fidelities_to_spend is not None
            or self.settings.max_evaluation_time_total_seconds is not None
        )

    def _write_trajectory_files(
        self,
        incumbent_configs: list,
        optimal_configs: list,
        trace_lock: FileLock,
        improvement_trace_path: Path,
        best_config_path: Path,
        final_stopping_criteria: ResourceUsage | None = None,
    ) -> None:
        """Writes the trajectory and best config files safely using generic file writer."""
        trace_text = _build_incumbent_content(incumbent_configs)

        best_config_text = _build_optimal_set_content(optimal_configs)

        if final_stopping_criteria:
            best_config_text += "\n" + "-" * 80
            best_config_text += "\nFinal cumulative metrics (Assuming completed run):"
            for metric, value in final_stopping_criteria.to_trajectory_dict().items():
                best_config_text += f"\n{metric}: {value}"

        with trace_lock:
            text_writer = get_file_writer("text")
            if incumbent_configs:
                try:
                    text_writer.write(trace_text, improvement_trace_path)
                except Exception as e:
                    logger.error(f"Failed to write improvement trace: {e}")
            if optimal_configs:
                try:
                    text_writer.write(best_config_text, best_config_path)
                except Exception as e:
                    logger.error(f"Failed to write best config: {e}")

    def _save_optimizer_artifacts(self, artifacts: list, summary_dir: Path) -> None:
        """Save optimizer artifacts to summary directory.
        
        Args:
            artifacts: List of Artifact objects to persist.
            summary_dir: Summary directory where artifacts will be saved.
        """
        logger.info("saving artifacts...")
        if artifacts is None:
            logger.warning("No artifacts found to save.")
            return

        for artifact in artifacts:
            try:
                # Map ArtifactType enum to string for writer lookup
                content_type = artifact.artifact_type.value
                writer = get_file_writer(content_type)
                file_path = summary_dir / artifact.name
                writer.write(artifact.content, file_path)
            except Exception as e:
                logger.error(
                    f"Failed to save artifact '{artifact.name}' "
                    f"(type={artifact.artifact_type.value}): {e}"
                )
                # Don't raise - allow optimization to continue even if artifact save fails
                continue

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
                    should_stop, stop_criteria = self._check_global_stopping_criterion(
                        trials,
                        log_status=True,
                    )
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
        summary_dir = main_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        improvement_trace_path = summary_dir / "best_config_trajectory.txt"
        improvement_trace_path.touch(exist_ok=True)
        best_config_path = summary_dir / "best_config.txt"
        best_config_path.touch(exist_ok=True)
        _trace_lock = FileLock(".trace.lock")
        _trace_lock_path = Path(str(_trace_lock.lock_file))
        _trace_lock_path.touch(exist_ok=True)
        full_df_path, short_path, csv_locker = _initiate_summary_csv(main_dir)

        # Create empty CSV files
        with csv_locker.lock():
            full_df_path.parent.mkdir(parents=True, exist_ok=True)
            full_df_path.touch(exist_ok=True)
            short_path.touch(exist_ok=True)

        logger.info(
            "Summary files can be found in the “summary” folder inside"
            " the root directory: %s",
            summary_dir,
        )

        optimizer_name = self.state._optimizer_info["name"]
        logger.info("Using optimizer: %s", optimizer_name)

        _time_monotonic_start = time.monotonic()
        _error_from_evaluation: Exception | None = None

        _repeated_fail_get_next_trial_count = 0
        n_repeated_failed_check_should_stop = 0

        evaluated_trials = self.state._trial_repo.get_valid_evaluated_trials()
        self.load_incumbent_trace(
            evaluated_trials,
            _trace_lock,
            improvement_trace_path,
            best_config_path,
        )

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

            if report is None:
                logger.info(
                    "Worker '%s' evaluated trial: %s async task detected.",
                    self.worker_id,
                    evaluated_trial.id,
                )
                continue

            logger.info(
                "Worker '%s' evaluated trial: %s as %s.",
                self.worker_id,
                evaluated_trial.id,
                evaluated_trial.metadata.state,
            )

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
                with self.state._trial_lock.lock():
                    evaluated_trials = self.state._trial_repo.get_valid_evaluated_trials()
                    self.load_incumbent_trace(
                        evaluated_trials,
                        _trace_lock,
                        improvement_trace_path,
                        best_config_path,
                    )
                            # Persist optimizer artifacts if available
                if hasattr(self.optimizer, 'get_trial_artifacts'):
                    try:
                        artifacts = self.optimizer.get_trial_artifacts(trials=evaluated_trials)
                        if artifacts is not None:
                            self._save_optimizer_artifacts(artifacts, summary_dir)
                    except Exception as e:
                        logger.error(f"Failed to persist optimizer artifacts: {e}", exc_info=True)
                

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

    def load_incumbent_trace(
        self,
        trials: dict[str, Trial],
        _trace_lock: FileLock,
        improvement_trace_path: Path,
        best_config_path: Path,
    ) -> None:
        """Load the incumbent trace from previous trials and update the state.
        This function also computes cumulative resource usage and updates the best
        configurations.

        Args:
            trials (dict): A dictionary of the evaluated trials which have a valid report.
            _trace_lock (FileLock): A file lock to ensure thread-safe writing.
            improvement_trace_path (Path): Path to the improvement trace file.
            best_config_path (Path): Path to the best configuration file.
        """
        if not trials:
            return

        # Clear any existing entries to prevent duplicates and rebuild a
        # non-dominated frontier from previous trials in chronological order.
        incumbent = []

        running_usage = ResourceUsage()

        sorted_trials: list[Trial] = sorted(
            trials.values(),
            key=lambda t: (
                t.metadata.time_sampled if t.metadata.time_sampled else float("inf")
            ),
        )
        is_mo = any(
            isinstance(trial.report.objective_to_minimize, list)  # type: ignore[union-attr]
            for trial in sorted_trials
        )

        frontier: list[Trial] = []
        trajectory_confs: dict[str, dict[str, float | int]] = {}

        for evaluated_trial in sorted_trials:
            single_trial_usage = self._calculate_total_resource_usage(
                {evaluated_trial.id: evaluated_trial}
            )
            running_usage += single_trial_usage

            assert evaluated_trial.report is not None  # for mypy
            new_trial_obj = evaluated_trial.report.objective_to_minimize

            if not _is_dominated(new_trial_obj, frontier):
                frontier = _prune_and_add_to_frontier(evaluated_trial, frontier)
                if not is_mo:
                    incumbent.append(evaluated_trial)
                current_snapshot = ResourceUsage(**asdict(running_usage))
                config_dict = {
                    "score": new_trial_obj,
                    "trial_id": evaluated_trial.id,
                    "config": evaluated_trial.config,
                }
                if evaluated_trial.report.cost is not None:
                    config_dict["cost"] = evaluated_trial.report.cost

                config_dict.update(current_snapshot.to_trajectory_dict())
                trajectory_confs[evaluated_trial.id] = config_dict

        optimal_configs: list[dict] = [trajectory_confs[trial.id] for trial in frontier]
        incumbent_configs: list[dict] = [
            trajectory_confs[trial.id] for trial in incumbent
        ]

        self._write_trajectory_files(
            incumbent_configs=incumbent_configs,
            optimal_configs=optimal_configs,
            trace_lock=_trace_lock,
            improvement_trace_path=improvement_trace_path,
            best_config_path=best_config_path,
        )


def _save_results(
    user_result: dict,
    trial_id: str,
    root_directory: Path,
) -> None:
    """Parse `user_result` and persist it for <trial_id> in the NePS state."""
    default_report_values = _make_default_report_values(
        objective_value_on_error=0, cost_value_on_error=0
    )

    result = UserResult.parse(
        user_result,
        default_cost_value=default_report_values.cost_if_not_provided,
        default_objective_to_minimize_value=default_report_values.objective_value_on_error,
        default_learning_curve=default_report_values.learning_curve_if_not_provided,
    )
    if result.exception is None and result.cost is None:
        logger.warning(
            "The return value of `evaluate_pipeline` "
            "must be a dictionary that includes a 'cost' key."
        )

    # load the NePS state from the optimization directory
    state = NePSState.create_or_load(path=root_directory, load_only=True)

    # lock the requested trial
    trial = state.lock_and_get_trial_by_id(trial_id)
    if trial is None:
        raise RuntimeError(f"Trial '{trial_id}' not found in '{root_directory}'")

    report = trial.set_complete(
        report_as=(
            Trial.State.SUCCESS.value
            if result.exception is None
            else Trial.State.CRASHED.value
        ),
        objective_to_minimize=result.objective_to_minimize,
        cost=result.cost,
        learning_curve=result.learning_curve,
        err=result.exception,
        tb=(
            "".join(
                traceback.format_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )
            )
            if result.exception is not None
            else None
        ),
        extra=result.extra,
        time_end=time.time(),
        evaluation_duration=result.cost,
    )

    worker_id = trial.metadata.evaluating_worker_id
    with state._trial_lock.lock():
        state._report_trial_evaluation(
            trial=trial,
            report=report,
            worker_id=worker_id,
        )
    for _, cb in _TRIAL_END_CALLBACKS.items():
        cb(trial)

    logger.info(f"Saved result for trial {trial.id}")


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
    pipeline_space: SearchSpace | PipelineSpace,
    cost_to_spend: float | None,
    ignore_errors: bool = False,
    objective_value_on_error: float | None,
    cost_value_on_error: float | None,
    continue_until_max_evaluation_completed: bool,
    overwrite_optimization_dir: bool,
    evaluations_to_spend: int | None,
    fidelities_to_spend: int | float | None,
    sample_batch_size: int | None,
    worker_id: str | None = None,
) -> None:
    default_report_values = _make_default_report_values(
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
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
                            cost_to_spend=cost_to_spend,
                            used_cost_budget=0,
                            max_evaluations=evaluations_to_spend,
                            fidelities_to_spend=fidelities_to_spend,
                            used_evaluations=0,
                        )
                    ),
                    shared_state=None,  # TODO: Unused for the time being...
                    worker_ids=None,
                ),
                pipeline_space=pipeline_space,
            )
            break
        except NePSError:
            # Don't retry on NePSError - these are user errors
            # like pipeline space mismatch
            raise
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
        evaluations_to_spend=evaluations_to_spend,
        fidelities_to_spend=fidelities_to_spend,
        include_in_progress_evaluations_towards_maximum=(
            not continue_until_max_evaluation_completed
        ),
        cost_to_spend=cost_to_spend,
        max_evaluation_time_total_seconds=None,  # TODO: User can't specify yet
        max_wallclock_time_seconds=None,  # TODO: User can't specify yet
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
        worker_id=worker_id,
    )
    worker.run()


def _make_default_report_values(
    *,
    objective_value_on_error: float | None = None,
    cost_value_on_error: float | None = None,
) -> DefaultReportValues:
    return DefaultReportValues(
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
        cost_if_not_provided=None,
        learning_curve_on_error=None,
        learning_curve_if_not_provided="objective_to_minimize",
    )


def _to_sequence(score: float | Sequence[float]) -> list[float]:
    """Normalize score to a list of floats for pareto comparisons.

    Scalars become single-element lists. Sequences are converted to lists.
    """
    if isinstance(score, Sequence):
        return [float(x) for x in score]
    return [float(score)]


def _is_dominated(candidate: float | Sequence[float], frontier: list[Trial]) -> bool:
    """Return True if `candidate` is dominated by any point in `frontier`.

    `frontier` is a list of score sequences (as lists).
    """
    cand_seq = _to_sequence(candidate)

    for t in frontier:
        if t.report is None:
            continue
        f_seq = _to_sequence(t.report.objective_to_minimize)
        if len(f_seq) != len(cand_seq):
            continue
        if all(fi <= ci for fi, ci in zip(f_seq, cand_seq, strict=False)) and any(
            fi < ci for fi, ci in zip(f_seq, cand_seq, strict=False)
        ):
            return True
    return False


def _prune_and_add_to_frontier(candidate: Trial, frontier: list[Trial]) -> list[Trial]:
    """Add candidate Trial to frontier and remove frontier Trials dominated by it.

    Frontier is a list of Trial objects (with reports). Returns the new frontier
    as a list of Trials.
    """
    if candidate.report is None:
        return frontier

    cand_seq = _to_sequence(candidate.report.objective_to_minimize)
    new_frontier: list[Trial] = []
    for t in frontier:
        if t.report is None:
            continue
        f_seq = _to_sequence(t.report.objective_to_minimize)
        if (
            len(f_seq) == len(cand_seq)
            and all(ci <= fi for ci, fi in zip(cand_seq, f_seq, strict=False))
            and any(ci < fi for ci, fi in zip(cand_seq, f_seq, strict=False))
        ):
            continue
        new_frontier.append(t)
    new_frontier.append(candidate)
    return new_frontier
