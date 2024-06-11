"""Module for the runtime of a single instance of NePS running.

An important advantage of NePS with a running instance per worker and no
multiprocessing is that we can reliably use globals to store information such
as the currently running configuration, without interfering with other
workers which have launched.

This allows us to have a global `Trial` object which can be accessed
using `import neps.runtime; neps.get_in_progress_trial()`.

---

This module primarily handles the worker loop where important concepts are:
* **State**: The state of optimization is all of the configurations, their results and
 the current state of the optimizer.
* **Shared State**: Whenever a worker wishes to read or write any state, they will _lock_
the shared state, declaring themselves as operating on it. At this point, no other worker
can access the shared state.
* **Optimizer Hydration**: This is the process through which an optimizer instance is
_hydrated_ with the Shared State so it can make a decision, i.e. for sampling.
Equally we _serialize_ the optimizer when writing it back to Shared State
* **Trial Lock**: When evaluating a configuration, a worker must _lock_ it to declared
itself as evaluating it. This communicates to other workers that this configuration is
in progress.

### Loop
We mark lines with `+` as the worker having locked the Shared State and `~` as the worker
having locked the Trial. The trial lock `~` is allowed to fail, in which case all steps
with a `~` are skipped and the loop continues.

1. + Check exit conditions
2. + Hydrate the optimizer
3. + Sample a new Trial
3. Unlock the Shared State
4. ~ Obtain a Trial Lock
5. ~ Set the global trial for this work to the current trial
6. ~ Evaluate the trial
7. ~+ Lock the shared state
8. ~+ Write the results of the config to disk
9. ~+ Update the optimizer if required (used budget for evaluating trial)
10. ~ Unlock the shared state
11. Unlock Trial Lock
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Union,
)
from typing_extensions import Self, TypeAlias

import numpy as np

from neps.utils._locker import Locker
from neps.utils._rng import SeedState
from neps.utils.files import deserialize, empty_file, serialize
from neps.utils.types import (
    ERROR,
    POST_EVAL_HOOK_SIGNATURE,
    ConfigID,
    ConfigResult,
    RawConfig,
)

if TYPE_CHECKING:
    from neps.optimizers.base_optimizer import BaseOptimizer
    from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger(__name__)

# Wait time between each successive poll to see if state can be grabbed
DEFAULT_STATE_POLL: float = 0.1
ENVIRON_STATE_POLL_KEY = "NEPS_STATE_POLL"

# Timeout before giving up on trying to grab the state, raising an error
DEFAULT_STATE_TIMEOUT: float | None = None
ENVIRON_STATE_TIMEOUT_KEY = "NEPS_STATE_TIMEOUT"


# TODO(eddiebergman): We should not do this...
warnings.simplefilter("always", DeprecationWarning)


# NOTE: As each NEPS process is only ever evaluating a single trial,
# this global can be retrieved in NePS and refers to what this process
# is currently evaluating.
_CURRENTLY_RUNNING_TRIAL_IN_PROCESS: Trial | None = None


def get_in_progress_trial() -> Trial | None:
    """Get the currently running trial in this process."""
    return _CURRENTLY_RUNNING_TRIAL_IN_PROCESS


def _set_in_progress_trial(trial: Trial | None) -> None:
    global _CURRENTLY_RUNNING_TRIAL_IN_PROCESS  # noqa: PLW0603
    _CURRENTLY_RUNNING_TRIAL_IN_PROCESS = trial


def get_shared_state_poll_and_timeout() -> tuple[float, float | None]:
    """Get the poll and timeout for the shared state."""
    poll = float(os.environ.get(ENVIRON_STATE_POLL_KEY, DEFAULT_STATE_POLL))
    timeout = os.environ.get(ENVIRON_STATE_TIMEOUT_KEY, DEFAULT_STATE_TIMEOUT)
    timeout = float(timeout) if timeout is not None else None
    return poll, timeout


@dataclass
class SuccessReport:
    """A successful report of the evaluation of a configuration."""

    loss: float
    cost: float | None
    account_for_cost: bool
    results: Mapping[str, Any]


@dataclass
class ErrorReport:
    """A failed report of the evaluation of a configuration."""

    err: Exception
    tb: str | None
    loss: float | None
    cost: float | None
    account_for_cost: bool
    results: Mapping[str, Any]


Report: TypeAlias = Union[SuccessReport, ErrorReport]


@dataclass
class Trial:
    """A trial is a configuration and it's associated data.

    The object is considered mutable and the global trial currently being
    evaluated can be access using `get_in_progress_trial()`.

    Attributes:
        id: Unique identifier for the configuration
        config: The configuration to evaluate
        pipeline_dir: Directory where the configuration is evaluated
        previous: The previous trial before this trial.
        time_sampled: The time the configuration was sampled
        metadata: Additional metadata about the configuration
    """

    id: ConfigID
    config: Mapping[str, Any]
    pipeline_dir: Path
    previous: Trial | None
    report: Report | None
    time_sampled: float
    metadata: dict[str, Any]
    _lock: Locker = field(init=False)
    disk: Trial.Disk = field(init=False)

    def to_config_result(
        self,
        config_to_search_space: Callable[[RawConfig], SearchSpace],
    ) -> ConfigResult:
        """Convert the report to a `ConfigResult` object."""
        result: ERROR | Mapping[str, Any] = (
            "error"
            if self.report is None or isinstance(self.report, ErrorReport)
            else self.report.results
        )
        return ConfigResult(
            self.id,
            config=config_to_search_space(self.config),
            result=result,
            metadata=self.metadata,
        )

    class State(Enum):
        """The state of a trial."""

        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        SUCCESS = "success"
        ERROR = "error"
        CORRUPTED = "corrupted"

    def __post_init__(self) -> None:
        if "time_sampled" not in self.metadata:
            self.metadata["time_sampled"] = self.time_sampled
        self.pipeline_dir.mkdir(exist_ok=True, parents=True)
        self._lock = Locker(self.pipeline_dir / ".config_lock")
        self.disk = Trial.Disk(pipeline_dir=self.pipeline_dir)

    @property
    def config_file(self) -> Path:
        """The path to the configuration file."""
        return self.pipeline_dir / "config.yaml"

    @property
    def metadata_file(self) -> Path:
        """The path to the metadata file."""
        return self.pipeline_dir / "metadata.yaml"

    @classmethod
    def from_dir(cls, pipeline_dir: Path, *, previous: Trial | None = None) -> Self:
        """Create a `Trial` object from a directory.

        Args:
            pipeline_dir: The directory where the trial is stored
            previous: The previous trial before this trial.
                You can use this to prevent loading the previous trial from disk,
                if it exists, i.e. a caching shortcut.

        Returns:
            The trial object.
        """
        return cls.from_disk(
            Trial.Disk.from_dir(pipeline_dir),
            previous=previous,
        )

    @classmethod
    def from_disk(cls, disk: Trial.Disk, *, previous: Trial | None = None) -> Self:
        """Create a `Trial` object from a disk.

        Args:
            disk: The disk information of the trial.
            previous: The previous trial before this trial.
                You can use this to prevent loading the previous trial from disk,
                if it exists, i.e. a caching shortcut.

        Returns:
            The trial object.
        """
        try:
            config = deserialize(disk.config_file)
        except Exception as e:
            logger.error(
                f"Error loading config from {disk.config_file}: {e}",
                exc_info=True,
            )
            config = {}

        try:
            metadata = deserialize(disk.metadata_file)
            time_sampled = metadata["time_sampled"]
        except Exception as e:
            logger.error(
                f"Error loading metadata from {disk.metadata_file}: {e}",
                exc_info=True,
            )
            metadata = {}
            time_sampled = float("nan")

        try:
            result: dict[str, Any] | tuple[Exception, str | None] | None
            report: Report | None
            if not empty_file(disk.result_file):
                result = deserialize(disk.result_file)

                assert isinstance(result, dict)
                report = SuccessReport(
                    loss=result["loss"],
                    cost=result.get("cost", None),
                    account_for_cost=result.get("account_for_cost", True),
                    results=result,
                )
            elif not empty_file(disk.error_file):
                error_tb = deserialize(disk.error_file)
                result = deserialize(disk.result_file)
                report = ErrorReport(
                    # NOTE: Not sure we can easily get the original exception type,
                    # once serialized
                    err=Exception(error_tb["err"]),
                    tb=error_tb.get("tb"),
                    loss=result.get("loss", None),
                    cost=result.get("cost", None),
                    account_for_cost=result.get("account_for_cost", True),
                    results=result,
                )
            else:
                report = None
        except Exception as e:
            logger.error(
                f"Error loading result from {disk.result_file}: {e}",
                exc_info=True,
            )
            report = None

        try:
            if previous is None and disk.previous_pipeline_dir is not None:
                previous = Trial.from_dir(disk.previous_pipeline_dir)
        except Exception as e:
            logger.error(
                f"Error loading previous from {disk.previous_pipeline_dir}: {e}",
                exc_info=True,
            )
            previous = None

        return cls(
            id=disk.config_id,
            config=config,
            pipeline_dir=disk.pipeline_dir,
            report=report,
            previous=previous,
            time_sampled=time_sampled,
            metadata=metadata,
        )

    @property
    def previous_config_id_file(self) -> Path:
        """The path to the previous configuration id file."""
        return self.pipeline_dir / "previous_config.id"

    def create_error_report(self, err: Exception, tb: str | None = None) -> ErrorReport:
        """Create a [`Report`][neps.runtime.Report] object with an error."""
        # TODO(eddiebergman): For now we assume the loss and cost for an error is None
        # and that we don't account for cost and there are no possible results.
        return ErrorReport(
            loss=None,
            cost=None,
            account_for_cost=False,
            results={},
            err=err,
            tb=tb,
        )

    def create_success_report(self, result: float | Mapping[str, Any]) -> SuccessReport:
        """Check if the trial has succeeded."""
        _result: dict[str, Any] = {}
        if isinstance(result, Mapping):
            if "loss" not in result:
                raise KeyError("The 'loss' should be provided in the evaluation result")

            _result = dict(result)
            loss = _result["loss"]
        else:
            loss = result

        try:
            _result["loss"] = float(loss)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "The evaluation result should be a dictionnary or a float but got"
                f" a `{type(loss)}` with value of {loss}",
            ) from e

        # TODO(eddiebergman): For now we have no access to the cost for crash
        # so we just set it to None.
        _cost: float | None = _result.get("cost", None)
        if _cost is not None:
            try:
                _result["cost"] = float(_cost)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The evaluation result should be a dictionnary or a float but got"
                    f" a `{type(_cost)}` with value of {_cost}",
                ) from e

        # TODO(eddiebergman): Should probably be a global user setting for this.
        _account_for_cost = _result.get("account_for_cost", True)

        return SuccessReport(
            loss=_result["loss"],
            cost=_cost,
            account_for_cost=_account_for_cost,
            results=_result,
        )

    @dataclass
    class Disk:
        """The disk information of a trial.

        Attributes:
            pipeline_dir: The directory where the trial is stored
            id: The unique identifier of the trial
            config_file: The path to the configuration file
            result_file: The path to the result file
            metadata_file: The path to the metadata file
            optimization_dir: The directory from which optimization is running
            previous_config_id_file: The path to the previous config id file
            previous_pipeline_dir: The directory of the previous configuration
            lock: The lock for the trial. Obtaining this lock indicates the worker
                is evaluating this trial.
        """

        pipeline_dir: Path

        config_id: ConfigID = field(init=False)
        config_file: Path = field(init=False)
        result_file: Path = field(init=False)
        error_file: Path = field(init=False)
        metadata_file: Path = field(init=False)
        optimization_dir: Path = field(init=False)
        previous_config_id_file: Path = field(init=False)
        previous_config_id: ConfigID | None = field(init=False)
        previous_pipeline_dir: Path | None = field(init=False)
        lock: Locker = field(init=False)

        def __post_init__(self) -> None:
            self.config_id = self.pipeline_dir.name[len("config_") :]
            self.config_file = self.pipeline_dir / "config.yaml"
            self.result_file = self.pipeline_dir / "result.yaml"
            self.error_file = self.pipeline_dir / "error.yaml"
            self.metadata_file = self.pipeline_dir / "metadata.yaml"

            # NOTE: This is a bit of an assumption!
            self.optimization_dir = self.pipeline_dir.parent

            self.previous_config_id_file = self.pipeline_dir / "previous_config.id"
            if not empty_file(self.previous_config_id_file):
                with self.previous_config_id_file.open("r") as f:
                    self.previous_config_id = f.read().strip()

                self.previous_pipeline_dir = (
                    self.pipeline_dir.parent / f"config_{self.previous_config_id}"
                )
            else:
                self.previous_pipeline_dir = None
                self.previous_config_id = None

            self.pipeline_dir.mkdir(exist_ok=True, parents=True)
            self.lock = Locker(self.pipeline_dir / ".config_lock")

        def raw_config(self) -> dict[str, Any]:
            """Deserialize the configuration from disk."""
            return deserialize(self.config_file)

        def state(self) -> Trial.State:  # noqa: PLR0911
            """The state of the trial."""
            result_file_exists = not empty_file(self.result_file)
            error_file_exists = not empty_file(self.error_file)
            config_file_exists = not empty_file(self.config_file)

            # NOTE: We don't handle the case where it's locked and there is a result
            # or error file existing, namely as this might introduce a race condition,
            # where the result/error is being written to while the lock still exists.

            if error_file_exists:
                # Should not have a results file if there is an error file
                if result_file_exists:
                    return Trial.State.CORRUPTED

                # Should have a config file if there is an error file
                if not config_file_exists:
                    return Trial.State.CORRUPTED

                return Trial.State.ERROR

            if result_file_exists:
                # Should have a config file if there is a results file
                if not config_file_exists:
                    return Trial.State.CORRUPTED

                return Trial.State.SUCCESS

            if self.lock.is_locked():
                # Should have a config to evaluate if it's locked
                if not config_file_exists:
                    return Trial.State.CORRUPTED

                return Trial.State.IN_PROGRESS

            return Trial.State.PENDING

        @classmethod
        def from_dir(cls, pipeline_dir: Path) -> Trial.Disk:
            """Create a `Trial.Disk` object from a directory."""
            return cls(pipeline_dir=pipeline_dir)


@dataclass
class StatePaths:
    """The paths used for the state of the optimization process.

    Most important method is [`config_dir`][neps.runtime.StatePaths.config_dir],
    which gives the directory to use for a configuration.

    Attributes:
        root: The root directory of the optimization process.
        create_dirs: Whether to create the directories if they do not exist.
        optimizer_state_file: The path to the optimizer state file.
        optimizer_info_file: The path to the optimizer info file.
        seed_state_dir: The directory where the seed state is stored.
        results_dir: The directory where results are stored.
    """

    root: Path
    create_dirs: bool = False

    optimizer_state_file: Path = field(init=False)
    optimizer_info_file: Path = field(init=False)
    seed_state_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.create_dirs:
            self.root.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.root / "results"

        if self.create_dirs:
            self.results_dir.mkdir(exist_ok=True)

        self.optimizer_state_file = self.root / ".optimizer_state.yaml"
        self.optimizer_info_file = self.root / ".optimizer_info.yaml"
        self.seed_state_dir = self.root / ".seed_state"

    def config_dir(self, config_id: ConfigID) -> Path:
        """Get the directory for a configuration."""
        return self.results_dir / f"config_{config_id}"


@dataclass
class SharedState:
    """The shared state of the optimization process that workers communicate through.

    Attributes:
        base_dir: The base directory from which the optimization is running.
        create_dirs: Whether to create the directories if they do not exist.
        lock: The lock to signify that a worker is operating on the shared state.
        optimizer_state_file: The path to the optimizers state.
        optimizer_info_file: The path to the file containing information about the
            optimizer's setup.
        seed_state_dir: Directory where the seed state is stored.
        results_dir: Directory where results for configurations are stored.
    """

    base_dir: Path
    paths: StatePaths = field(init=False)
    create_dirs: bool = False
    lock: Locker = field(init=False)

    trials: dict[ConfigID, tuple[Trial, Trial.State]] = field(default_factory=dict)
    """Mapping from a configid to the trial and it's last known state, including if
    it's been evaluated."""

    def __post_init__(self) -> None:
        self.paths = StatePaths(root=self.base_dir, create_dirs=self.create_dirs)
        self.lock = Locker(self.base_dir / ".decision_lock")

    def trials_by_state(self) -> dict[Trial.State, list[Trial]]:
        """Get the trials grouped by their state."""
        _dict: dict[Trial.State, list[Trial]] = {state: [] for state in Trial.State}
        for trial, state in self.trials.values():
            _dict[state].append(trial)
        return _dict

    def check_optimizer_info_on_disk_matches(
        self,
        optimizer_info: dict[str, Any],
        *,
        excluded_keys: Iterable[str] = ("searcher_name",),
    ) -> None:
        """Sanity check that the provided info matches the one on disk (if any).

        Args:
            optimizer_info: The optimizer info to check.
            excluded_keys: Any keys to exclude during the comparison.

        Raises:
            ValueError: If there is optimizer info on disk and it does not match the
            provided info.
        """
        optimizer_info_copy = optimizer_info.copy()
        loaded_info = deserialize(self.paths.optimizer_info_file)

        for key in excluded_keys:
            optimizer_info_copy.pop(key, None)
            loaded_info.pop(key, None)

        if optimizer_info_copy != loaded_info:
            raise ValueError(
                f"The sampler_info in the file {self.paths.optimizer_info_file} is not"
                f" valid. Expected: {optimizer_info_copy}, Found: {loaded_info}",
            )

    @contextmanager
    def use_sampler(
        self,
        sampler: BaseOptimizer,
        *,
        serialize_seed: bool = True,
    ) -> Iterator[BaseOptimizer]:
        """Use the sampler with the shared state."""
        if serialize_seed:
            with SeedState.use(self.paths.seed_state_dir), sampler.using_state(
                self.paths.optimizer_state_file
            ):
                yield sampler
        else:
            with sampler.using_state(self.paths.optimizer_state_file):
                yield sampler

    def update_from_disk(self) -> None:
        """Update the shared state from disk."""
        trial_dirs = (p for p in self.paths.results_dir.iterdir() if p.is_dir())
        _disks = [Trial.Disk.from_dir(p) for p in trial_dirs]
        _disk_lookup = {disk.config_id: disk for disk in _disks}

        # NOTE: We sort all trials such that we process previous trials first, i.e.
        # if trial_3 has trial_2 as previous, we process trial_2 first, which
        # requires trial_1 to have been processed first.
        def _depth(trial: Trial.Disk) -> int:
            depth = 0
            previous = trial.previous_config_id
            while previous is not None:
                depth += 1
                previous_trial = _disk_lookup.get(previous)
                if previous_trial is None:
                    raise RuntimeError(
                        "Previous trial not found on disk when processing a trial."
                        " This should not happen as if a tria has a previous trial,"
                        " then it should be present and evaluated on disk.",
                    )
                previous = previous_trial.previous_config_id

            return depth

        # This allows is to traverse linearly and used cached values of previous
        # trial data loading, as done below.
        _disks.sort(key=_depth)

        for disk in _disks:
            config_id = disk.config_id
            state = disk.state()

            if state is Trial.State.CORRUPTED:
                logger.warning(f"Trial {config_id} was corrupted somehow!")

            previous: Trial | None = None
            if disk.previous_config_id is not None:
                previous, _ = self.trials.get(disk.previous_config_id, (None, None))
                if previous is None:
                    raise RuntimeError(
                        "Previous trial not found in memory when processing a trial."
                        " This should not happen as if a trial has a previous trial,"
                        " then it should be present and evaluated in memory.",
                    )

            cached_trial = self.trials.get(config_id, None)

            # If not currently cached or it was and had a state change
            if cached_trial is None or cached_trial[1] != state:
                trial = Trial.from_disk(disk, previous=previous)
                self.trials[config_id] = (trial, state)

    @contextmanager
    def sync(self, *, lock: bool = True) -> Iterator[None]:
        """Sync up with what's on disk."""
        if lock:
            _poll, _timeout = get_shared_state_poll_and_timeout()
            with self.lock(poll=_poll, timeout=_timeout):
                self.update_from_disk()
                yield
        else:
            yield


def _evaluate_config(
    trial: Trial,
    evaluation_fn: Callable[..., float | Mapping[str, Any]],
    logger: logging.Logger,
) -> float | Mapping[str, Any]:
    config = trial.config
    config_id = trial.id
    pipeline_directory = trial.pipeline_dir
    previous_pipeline_directory = (
        None if trial.previous is None else trial.previous.pipeline_dir
    )

    logger.info(f"Start evaluating config {config_id}")

    # If pipeline_directory and previous_pipeline_directory are included in the
    # signature we supply their values, otherwise we simply do nothing.
    directory_params: list[Path | None] = []

    evaluation_fn_params = inspect.signature(evaluation_fn).parameters
    if "pipeline_directory" in evaluation_fn_params:
        directory_params.append(pipeline_directory)
    if "previous_pipeline_directory" in evaluation_fn_params:
        directory_params.append(previous_pipeline_directory)

    return evaluation_fn(*directory_params, **config)


def _worker_should_continue(
    max_evaluations_total: int | None,
    *,
    n_inprogress: int,
    n_evaluated: int,
    continue_until_max_evaluation_completed: bool,
) -> bool:
    # Check if we have reached the total amount of configurations to evaluated
    # (including pending evaluations possibly)
    if max_evaluations_total is None:
        return True

    n_counter = (
        n_evaluated
        if continue_until_max_evaluation_completed
        else n_evaluated + n_inprogress
    )
    return n_counter < max_evaluations_total


def _sample_trial_from_optimizer(
    optimizer: BaseOptimizer,
    config_dir_f: Callable[[ConfigID], Path],
    evaluated_trials: Mapping[ConfigID, Trial],
    pending_trials: Mapping[ConfigID, Trial],
) -> Trial:
    optimizer.load_results(
        previous_results={
            config_id: report.to_config_result(optimizer.load_config)
            for config_id, report in evaluated_trials.items()
        },
        pending_evaluations={
            config_id: optimizer.load_config(trial.config)
            for config_id, trial in pending_trials.items()
        },
    )
    config, config_id, prev_config_id = optimizer.get_config_and_ids()
    previous = None
    if prev_config_id is not None:
        previous = evaluated_trials[prev_config_id]

    time_sampled = time.time()
    return Trial(
        id=config_id,
        config=config,
        report=None,
        time_sampled=time_sampled,
        pipeline_dir=config_dir_f(config_id),
        previous=previous,
        metadata={"time_sampled": time_sampled},
    )


def launch_runtime(  # noqa: PLR0913, C901, PLR0915
    *,
    evaluation_fn: Callable[..., float | Mapping[str, Any]],
    sampler: BaseOptimizer,
    optimizer_info: dict,
    optimization_dir: Path | str,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    logger: logging.Logger | None = None,
    post_evaluation_hook: POST_EVAL_HOOK_SIGNATURE | None = None,
    overwrite_optimization_dir: bool = False,
    pre_load_hooks: Iterable[Callable[[BaseOptimizer], BaseOptimizer]] | None = None,
) -> None:
    """Launch the runtime of a single instance of NePS.

    Please refer to the module docstring for a detailed explanation of the runtime.
    Runs until some exit condition is met.

    Args:
        evaluation_fn: The evaluation function to use.
        sampler: The optimizer to use for sampling configurations.
        optimizer_info: Information about the optimizer.
        optimization_dir: The directory where the optimization is running.
        max_evaluations_total: The maximum number of evaluations to run.
        max_evaluations_per_run: The maximum number of evaluations to run in a single run.
        continue_until_max_evaluation_completed: Whether to continue until the maximum
            evaluations are completed.
        logger: The logger to use.
        post_evaluation_hook: A hook to run after the evaluation.
        overwrite_optimization_dir: Whether to overwrite the optimization directory.
        pre_load_hooks: Hooks to run before loading the results.
    """
    # NOTE(eddiebergman): This was deprecated a while ago and called at
    # evaluate, now we just crash immediatly instead. Should probably
    # promote this check closer to the user, i.e. `neps.run()`
    evaluation_fn_params = inspect.signature(evaluation_fn).parameters
    if "previous_working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'previous_working_directory' was deprecated. "
            f"In the function: '{evaluation_fn.__name__}', please,  "
            "use 'previous_pipeline_directory' instead. ",
        )
    if "working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'working_directory' was deprecated. "
            f"In the function: '{evaluation_fn.__name__}', please,  "
            "use 'pipeline_directory' instead. ",
        )

    if logger is None:
        logger = logging.getLogger("neps")

    optimization_dir = Path(optimization_dir)

    # TODO(eddiebergman): Not sure how overwriting works with multiple workers....
    if overwrite_optimization_dir and optimization_dir.exists():
        logger.warning("Overwriting working_directory")
        shutil.rmtree(optimization_dir)

    shared_state = SharedState(optimization_dir, create_dirs=True)

    _poll, _timeout = get_shared_state_poll_and_timeout()
    with shared_state.sync(lock=True):
        if not shared_state.paths.optimizer_info_file.exists():
            serialize(
                optimizer_info,
                shared_state.paths.optimizer_info_file,
                sort_keys=False,
            )
        else:
            shared_state.check_optimizer_info_on_disk_matches(optimizer_info)

    _max_evals_this_run = (
        max_evaluations_per_run if max_evaluations_per_run is not None else np.inf
    )

    evaluations_in_this_run = 0
    while True:
        if evaluations_in_this_run >= _max_evals_this_run:
            logger.info("Maximum evaluations per run is reached, shutting down")
            break

        with shared_state.sync(lock=True):
            trials_by_state = shared_state.trials_by_state()
            if not _worker_should_continue(
                max_evaluations_total,
                n_inprogress=len(trials_by_state[Trial.State.IN_PROGRESS]),
                n_evaluated=(
                    len(trials_by_state[Trial.State.SUCCESS])
                    + len(trials_by_state[Trial.State.ERROR])
                ),
                continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
            ):
                logger.info("Maximum total evaluations is reached, shutting down")
                break

            # While we have the decision lock, we will now sample
            # with the optimizer in this process
            with shared_state.use_sampler(sampler) as sampler:
                if sampler.is_out_of_budget():
                    logger.info("Maximum budget reached, shutting down")
                    break

                if pre_load_hooks is not None:
                    for hook in pre_load_hooks:
                        sampler = hook(sampler)  # noqa: PLW2901

                logger.debug("Sampling a new configuration")

                evaluated = (
                    trials_by_state[Trial.State.SUCCESS]
                    + trials_by_state[Trial.State.ERROR]
                )
                pending = (
                    trials_by_state[Trial.State.PENDING]
                    + trials_by_state[Trial.State.IN_PROGRESS]
                )
                trial = _sample_trial_from_optimizer(
                    sampler,
                    shared_state.paths.config_dir,
                    evaluated_trials={trial.id: trial for trial in evaluated},
                    pending_trials={trial.id: trial for trial in pending},
                )
                serialize(trial.config, trial.config_file)
                serialize(trial.metadata, trial.metadata_file)
                if trial.previous is not None:
                    trial.previous_config_id_file.write_text(trial.previous.id)

            logger.debug(f"Sampled config {trial.id}")

        # Obtain the lock on this trial and evaluate it,
        # otherwise continue back to waiting to sampling
        with trial._lock.try_lock() as acquired:
            if not acquired:
                continue

            # Inform the global state that this trial is being evaluated
            _set_in_progress_trial(trial)

            # TODO(eddiebergman): Right now if a trial crashes, it's cost is not accounted
            # for, this should probably removed from BaseOptimizer as it does not need
            # to know this and the runtime can fill this in for it.
            try:
                user_result = _evaluate_config(trial, evaluation_fn, logger)
            except Exception as e:  # noqa: BLE001
                # TODO(eddiebergman): Right now this never accounts for cost!
                # NOTE: It's important to lock the shared state such that any
                # sampling done is with taking this result into account
                # accidentally reads this config as un-evaluated
                with shared_state.lock(poll=_poll, timeout=_timeout):
                    # TODO(eddiebergman): We should add an option to just crash here
                    # if something goes wrong and raise up this error to the top.
                    logger.error(
                        f"Error during evaluation of '{trial.id}': {trial.config}."
                    )
                    logger.exception(e)
                    tb = traceback.format_exc()

                    trial.report = trial.create_error_report(e, tb=tb)
                    trial.metadata["time_end"] = time.time()

                    shared_state.trials[trial.id] = (trial, Trial.State.ERROR)

                    serialize({"err": str(e), "tb": tb}, trial.disk.error_file)
                    serialize(trial.metadata, trial.disk.metadata_file)
            else:
                trial.report = trial.create_success_report(user_result)
                trial.metadata["time_end"] = time.time()
                if sampler.budget is not None and trial.report.cost is None:
                    raise ValueError(
                        "The evaluation function result should contain a 'cost'"
                        f"field when used with a budget. Got {trial.report.results}",
                    )

                with shared_state.lock(poll=_poll, timeout=_timeout):
                    shared_state.trials[trial.id] = (trial, Trial.State.SUCCESS)

                    eval_cost = trial.report.cost
                    account_for_cost = False
                    if eval_cost is not None:
                        account_for_cost = trial.report.account_for_cost
                        budget_metadata = {
                            "max": sampler.budget,
                            "used": sampler.used_budget,
                            "eval_cost": eval_cost,
                            "account_for_cost": account_for_cost,
                        }
                        trial.metadata.update(budget_metadata)

                    serialize(trial.metadata, trial.disk.metadata_file)
                    serialize(trial.report.results, trial.disk.result_file)
                    if account_for_cost:
                        assert eval_cost is not None
                        with shared_state.use_sampler(sampler, serialize_seed=False):
                            sampler.used_budget += eval_cost

            _result: ERROR | dict[str, Any]
            if post_evaluation_hook is not None:
                report = trial.report
                if isinstance(report, ErrorReport):
                    _result = "error"
                elif isinstance(report, SuccessReport):
                    _result = dict(report.results)
                else:
                    _type = type(report)
                    raise TypeError(f"Unknown result type '{_type}' for report: {report}")

                post_evaluation_hook(
                    trial.config,
                    trial.id,
                    trial.pipeline_dir,
                    _result,
                    logger,
                )

            evaluations_in_this_run += 1
            logger.info(f"Finished evaluating config {trial.id}")
