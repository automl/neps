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
from copy import deepcopy
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
from typing_extensions import TypeAlias

import numpy as np

from neps.types import (
    ERROR,
    POST_EVAL_HOOK_SIGNATURE,
    ConfigID,
    ConfigResult,
    Metadata,
    RawConfig,
)
from neps.utils._locker import Locker
from neps.utils._rng import SeedState
from neps.utils.files import deserialize, empty_file, serialize

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
    """A successful report of the evaluation of a configuration.

    Attributes:
        trial: The trial that was evaluated
        id: The identifier of the configuration
        loss: The loss of the evaluation
        cost: The cost of the evaluation
        results: The results of the evaluation
        config: The configuration that was evaluated
        metadata: Additional metadata about the configuration
        pipeline_dir: The directory where the configuration was evaluated
        previous: The report of the previous iteration of this trial
        time_sampled: The time the configuration was sampled
        time_end: The time the configuration was evaluated
    """

    trial: Trial
    id: ConfigID
    loss: float
    cost: float | None
    account_for_cost: bool
    results: Mapping[str, Any]
    config: RawConfig
    metadata: Metadata
    pipeline_dir: Path
    previous: Report | None  # NOTE: Assumption only one previous report
    time_sampled: float
    time_end: float

    def __post_init__(self) -> None:
        if "time_end" not in self.metadata:
            self.metadata["time_end"] = self.time_end
        if "time_sampled" not in self.metadata:
            self.metadata["time_sampled"] = self.time_sampled

    @property
    def disk(self) -> TrialOnDisk:
        """Access the disk information of the trial."""
        return TrialOnDisk(pipeline_dir=self.pipeline_dir)

    def to_config_result(
        self,
        config_to_search_space: Callable[[RawConfig], SearchSpace],
    ) -> ConfigResult:
        """Convert the report to a `ConfigResult` object."""
        return ConfigResult(
            self.id,
            config=config_to_search_space(self.config),
            result=self.results,
            metadata=self.metadata,
        )


@dataclass
class ErrorReport:
    """A failed report of the evaluation of a configuration.

    Attributes:
        trial: The trial that was evaluated
        id: The identifier of the configuration
        loss: The loss of the evaluation, if any
        cost: The cost of the evaluation, if any
        results: The results of the evaluation
        config: The configuration that was evaluated
        metadata: Additional metadata about the configuration
        pipeline_dir: The directory where the configuration was evaluated
        previous: The report of the previous iteration of this trial
        time_sampled: The time the configuration was sampled
        time_end: The time the configuration was evaluated
    """

    trial: Trial
    id: ConfigID
    err: Exception
    tb: str | None
    loss: float | None
    cost: float | None
    account_for_cost: bool
    results: Mapping[str, Any]
    config: RawConfig
    metadata: Metadata
    pipeline_dir: Path
    previous: Report | None  # NOTE: Assumption only one previous report
    time_sampled: float
    time_end: float

    def __post_init__(self) -> None:
        if "time_end" not in self.metadata:
            self.metadata["time_end"] = self.time_end
        if "time_sampled" not in self.metadata:
            self.metadata["time_sampled"] = self.time_sampled

    @property
    def disk(self) -> TrialOnDisk:
        """Access the disk information of the trial."""
        return TrialOnDisk(pipeline_dir=self.pipeline_dir)

    def to_config_result(
        self,
        config_to_search_space: Callable[[RawConfig], SearchSpace],
    ) -> ConfigResult:
        """Convert the report to a `ConfigResult` object."""
        return ConfigResult(
            self.id,
            config=config_to_search_space(self.config),
            result="error",
            metadata=self.metadata,
        )


Report: TypeAlias = Union[SuccessReport, ErrorReport]


@dataclass
class TrialOnDisk:
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

    id: ConfigID = field(init=False)
    config_file: Path = field(init=False)
    result_file: Path = field(init=False)
    error_file: Path = field(init=False)
    metadata_file: Path = field(init=False)
    optimization_dir: Path = field(init=False)
    previous_config_id_file: Path = field(init=False)
    previous_pipeline_dir: Path | None = field(init=False)
    lock: Locker = field(init=False)

    def __post_init__(self) -> None:
        self.id = self.pipeline_dir.name[len("config_") :]
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
        self.pipeline_dir.mkdir(exist_ok=True, parents=True)
        self.lock = Locker(self.pipeline_dir / ".config_lock")

    class State(Enum):
        """The state of a trial."""

        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        SUCCESS = "success"
        ERROR = "error"
        CORRUPTED = "corrupted"

    def raw_config(self) -> dict[str, Any]:
        """Deserialize the configuration from disk."""
        return deserialize(self.config_file)

    def state(self) -> TrialOnDisk.State:
        """The state of the trial."""
        if not empty_file(self.result_file):
            return TrialOnDisk.State.SUCCESS
        if not empty_file(self.error_file):
            return TrialOnDisk.State.ERROR
        if self.lock.is_locked():
            return TrialOnDisk.State.IN_PROGRESS
        if not empty_file(self.config_file):
            return TrialOnDisk.State.PENDING

        return TrialOnDisk.State.CORRUPTED

    @classmethod
    def from_dir(cls, pipeline_dir: Path) -> TrialOnDisk:
        """Create a `Trial.Disk` object from a directory."""
        return cls(pipeline_dir=pipeline_dir)

    def load(
        self,
    ) -> tuple[
        RawConfig,
        Metadata,
        ConfigID | None,
        dict[str, Any] | tuple[Exception, str | None] | None,
    ]:
        """Load the trial from disk."""
        config = deserialize(self.config_file)
        metadata = deserialize(self.metadata_file)

        result: dict[str, Any] | tuple[Exception, str | None] | None
        if not empty_file(self.result_file):
            result = deserialize(self.result_file)
            assert isinstance(result, dict)
        elif not empty_file(self.error_file):
            error_tb = deserialize(self.error_file)
            result = (Exception(error_tb["err"]), error_tb.get("tb"))
        else:
            result = None

        if not empty_file(self.previous_config_id_file):
            previous_config_id = self.previous_config_id_file.read_text().strip()
        else:
            previous_config_id = None

        return config, metadata, previous_config_id, result


@dataclass
class Trial:
    """A trial is a configuration and it's associated data.

    The object is considered mutable and the global trial currently being
    evaluated can be access using `get_in_progress_trial()`.

    Attributes:
        id: Unique identifier for the configuration
        config: The configuration to evaluate
        pipeline_dir: Directory where the configuration is evaluated
        previous: The report of the previous iteration of this trial
        time_sampled: The time the configuration was sampled
        metadata: Additional metadata about the configuration
    """

    id: ConfigID
    config: Mapping[str, Any]
    pipeline_dir: Path
    previous: Report | None
    time_sampled: float
    metadata: dict[str, Any]
    _lock: Locker = field(init=False)
    disk: TrialOnDisk = field(init=False)

    def __post_init__(self) -> None:
        if "time_sampled" not in self.metadata:
            self.metadata["time_sampled"] = self.time_sampled
        self.pipeline_dir.mkdir(exist_ok=True, parents=True)
        self._lock = Locker(self.pipeline_dir / ".config_lock")
        self.disk = TrialOnDisk(pipeline_dir=self.pipeline_dir)

    @property
    def config_file(self) -> Path:
        """The path to the configuration file."""
        return self.pipeline_dir / "config.yaml"

    @property
    def metadata_file(self) -> Path:
        """The path to the metadata file."""
        return self.pipeline_dir / "metadata.yaml"

    @property
    def previous_config_id_file(self) -> Path:
        """The path to the previous configuration id file."""
        return self.pipeline_dir / "previous_config.id"

    def error(
        self,
        err: Exception,
        tb: str | None = None,
        *,
        time_end: float | None = None,
    ) -> ErrorReport:
        """Create a [`Report`][neps.runtime.Report] object with an error."""
        time_end = time_end if time_end is not None else time.time()
        if time_end not in self.metadata:
            self.metadata["time_end"] = time_end

        # TODO(eddiebergman): For now we assume the loss and cost for an error is None
        # and that we don't account for cost and there are no possible results.
        return ErrorReport(
            config=self.config,
            id=self.id,
            loss=None,
            cost=None,
            account_for_cost=False,
            results={},
            err=err,
            tb=tb,
            pipeline_dir=self.pipeline_dir,
            previous=self.previous,
            trial=self,
            metadata=self.metadata,
            time_sampled=self.time_sampled,
            time_end=time_end,
        )

    def success(
        self,
        result: float | Mapping[str, Any],
        *,
        time_end: float | None = None,
    ) -> SuccessReport:
        """Check if the trial has succeeded."""
        time_end = time_end if time_end is not None else time.time()
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
            config=self.config,
            id=self.id,
            loss=_result["loss"],
            cost=_cost,
            account_for_cost=_account_for_cost,
            results=_result,
            pipeline_dir=self.pipeline_dir,
            previous=self.previous,
            trial=self,
            metadata=self.metadata,
            time_sampled=self.time_sampled,
            time_end=time_end,
        )


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
    evaluated_trials: dict[ConfigID, Report] = field(default_factory=dict)
    pending_trials: dict[ConfigID, Trial] = field(default_factory=dict)
    in_progress_trials: dict[ConfigID, Trial] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.paths = StatePaths(root=self.base_dir, create_dirs=self.create_dirs)
        self.lock = Locker(self.base_dir / ".decision_lock")

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

    def update_from_disk(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Update the shared state from disk."""
        trial_dirs = (p for p in self.paths.results_dir.iterdir() if p.is_dir())
        trials_on_disk = [TrialOnDisk.from_dir(p) for p in trial_dirs]

        for trial_on_disk in trials_on_disk:
            state = trial_on_disk.state()

            if state in (TrialOnDisk.State.SUCCESS, TrialOnDisk.State.ERROR):
                if trial_on_disk.id in self.evaluated_trials:
                    continue

                # It's been evaluated and we can move it out of pending
                self.pending_trials.pop(trial_on_disk.id, None)
                self.in_progress_trials.pop(trial_on_disk.id, None)

                raw_config, metadata, previous_config_id, result = trial_on_disk.load()

                # NOTE: Assuming that the previous one will always have been
                # evaluated, if there is a previous one.
                previous_report = None
                if previous_config_id is not None:
                    previous_report = self.evaluated_trials[previous_config_id]

                trial = Trial(
                    id=trial_on_disk.id,
                    config=raw_config,
                    pipeline_dir=trial_on_disk.pipeline_dir,
                    previous=previous_report,
                    time_sampled=metadata["time_sampled"],
                    metadata=metadata,
                )

                report: Report
                if isinstance(result, dict):
                    report = trial.success(result, time_end=metadata["time_end"])
                elif isinstance(result, tuple):
                    err, tb = result
                    report = trial.error(err=err, tb=tb, time_end=metadata["time_end"])
                elif result is None:
                    raise RuntimeError(
                        "Result should not have been None, this is a bug!",
                        "Please report this to the developers with some sample code"
                        " if possible.",
                    )
                else:
                    raise TypeError(f"Unknown result type {type(result)}")

                self.evaluated_trials[trial_on_disk.id] = report

            elif state is TrialOnDisk.State.PENDING:
                assert trial_on_disk.id not in self.evaluated_trials
                if trial_on_disk.id in self.pending_trials:
                    continue

                raw_config, metadata, previous_config_id, result = trial_on_disk.load()

                # NOTE: Assuming that the previous one will always have been evaluated,
                # if there is a previous one.
                previous_report = None
                if previous_config_id is not None:
                    previous_report = self.evaluated_trials[previous_config_id]

                trial = Trial(
                    id=trial_on_disk.id,
                    config=raw_config,
                    pipeline_dir=trial_on_disk.pipeline_dir,
                    previous=previous_report,
                    time_sampled=metadata["time_sampled"],
                    metadata=metadata,
                )
                self.pending_trials[trial_on_disk.id] = trial

            elif state is TrialOnDisk.State.IN_PROGRESS:
                assert trial_on_disk.id not in self.evaluated_trials
                if trial_on_disk.id in self.in_progress_trials:
                    continue

                # If this was previously in the pending queue, jsut pop
                # it into the in progress queue
                previously_pending_trial = self.pending_trials.pop(trial_on_disk.id, None)
                if previously_pending_trial is not None:
                    self.in_progress_trials[trial_on_disk.id] = previously_pending_trial
                    continue

                # Otherwise it's the first time we saw it so we have to load it in
                raw_config, metadata, previous_config_id, result = trial_on_disk.load()

                # NOTE: Assuming that the previous one will always have been evaluated,
                # if there is a previous one.
                previous_report = None
                if previous_config_id is not None:
                    previous_report = self.evaluated_trials[previous_config_id]

                trial = Trial(
                    id=trial_on_disk.id,
                    config=raw_config,
                    pipeline_dir=trial_on_disk.pipeline_dir,
                    previous=previous_report,
                    time_sampled=metadata["time_sampled"],
                    metadata=metadata,
                )
                self.pending_trials[trial_on_disk.id] = trial

            elif state == TrialOnDisk.State.CORRUPTED:
                logger.warning(f"Removing corrupted trial {trial_on_disk.id}")
                try:
                    shutil.rmtree(trial_on_disk.pipeline_dir)
                except Exception as e:
                    logger.exception(e)

            else:
                raise ValueError(f"Unknown state {state} for trial {trial_on_disk.id}")

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

    config = deepcopy(config)

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
    evaluated_trials: Mapping[ConfigID, Report],
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
            if not _worker_should_continue(
                max_evaluations_total,
                n_inprogress=len(shared_state.pending_trials),
                n_evaluated=len(shared_state.evaluated_trials),
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

                trial = _sample_trial_from_optimizer(
                    sampler,
                    shared_state.paths.config_dir,
                    evaluated_trials=shared_state.evaluated_trials,
                    pending_trials=shared_state.pending_trials,
                )
                serialize(trial.config, trial.config_file)
                serialize(trial.metadata, trial.metadata_file)
                if trial.previous is not None:
                    trial.previous_config_id_file.write_text(trial.previous.trial.id)

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
            report: Report
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
                    report = trial.error(e, tb=tb, time_end=time.time())
                    serialize({"err": str(e), "tb": tb}, report.disk.error_file)
                    serialize(report.metadata, report.disk.metadata_file)
            else:
                report = trial.success(user_result, time_end=time.time())
                if sampler.budget is not None and report.cost is None:
                    raise ValueError(
                        "The evaluation function result should contain "
                        f"a 'cost' field when used with a budget. Got {report.results}",
                    )

                with shared_state.lock(poll=_poll, timeout=_timeout):
                    eval_cost = report.cost
                    account_for_cost = False
                    if eval_cost is not None:
                        account_for_cost = report.account_for_cost
                        budget_metadata = {
                            "max": sampler.budget,
                            "used": sampler.used_budget,
                            "eval_cost": eval_cost,
                            "account_for_cost": account_for_cost,
                        }
                        trial.metadata.update(budget_metadata)

                    serialize(report.metadata, report.disk.metadata_file)
                    serialize(report.results, report.disk.result_file)
                    if account_for_cost:
                        assert eval_cost is not None
                        with shared_state.use_sampler(sampler, serialize_seed=False):
                            sampler.used_budget += eval_cost

            _result: ERROR | dict[str, Any]
            if post_evaluation_hook is not None:
                if isinstance(report, ErrorReport):
                    _result = "error"
                elif isinstance(report, SuccessReport):
                    _result = dict(report.results)
                else:
                    raise TypeError(
                        f"Unknown result type '{type(report)}' for report: {report}"
                    )
                post_evaluation_hook(
                    trial.config,
                    trial.id,
                    trial.pipeline_dir,
                    _result,
                    logger,
                )

            evaluations_in_this_run += 1
            logger.info(f"Finished evaluating config {trial.id}")
