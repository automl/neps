"""The main state object that holds all the shared state objects.

This object is used to interact with the shared state objects in a safe atomic
manner, such that each worker can create an identical NePSState and interact with
it without having to worry about locking or out-dated information.

For an actual instantiation of this object, see
[`create_or_load_filebased_neps_state`][neps.state.filebased.create_or_load_filebased_neps_state].
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)
from uuid import uuid4

from neps.env import (
    STATE_FILELOCK_POLL,
    STATE_FILELOCK_TIMEOUT,
    TRIAL_FILELOCK_POLL,
    TRIAL_FILELOCK_TIMEOUT,
)
from neps.exceptions import NePSError, TrialAlreadyExistsError, TrialNotFoundError
from neps.state.err_dump import ErrDump
from neps.state.filebased import (
    FileLocker,
    ReaderWriterErrDump,
    ReaderWriterOptimizationState,
    ReaderWriterOptimizerInfo,
    ReaderWriterSeedSnapshot,
    TrialReaderWriter,
    TrialWriteHint,
)
from neps.state.optimizer import OptimizationState, OptimizerInfo
from neps.state.seed_snapshot import SeedSnapshot
from neps.state.trial import Report, Trial

if TYPE_CHECKING:
    from neps.optimizers.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)
N_UNSAFE_RETRIES = 10

# TODO: Technically we don't need the same Location type for all shared objects.
Loc = TypeVar("Loc")
T = TypeVar("T")

Version: TypeAlias = str

Resource: TypeAlias = Literal[
    "optimizer_info", "optimizer_state", "seed_state", "errors", "configs"
]


def make_sha() -> Version:
    """Generate a str hex sha."""
    return uuid4().hex


CONFIG_PREFIX_LEN = len("config_")


# TODO: Ergonomics of this class sucks
@dataclass
class TrialRepo:
    CACHE_FILE_NAME = ".trial_cache.pkl"

    directory: Path
    cache_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.directory / self.CACHE_FILE_NAME

    def list_trial_ids(self) -> list[str]:
        return [
            config_path.name[CONFIG_PREFIX_LEN:]
            for config_path in self.directory.iterdir()
            if config_path.name.startswith("config_") and config_path.is_dir()
        ]

    def latest(self) -> dict[str, Trial]:
        if not self.cache_path.exists():
            # If we end up with no cache but there are trials on disk, we need to
            # read them in. However we will not save back the cache here in fear of
            # overwriting
            if any(path.name.startswith("config_") for path in self.directory.iterdir()):
                trial_ids = self.list_trial_ids()
                return {
                    trial_id: self.load_trial_from_disk(trial_id)
                    for trial_id in trial_ids
                }

            return {}

        with self.cache_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def new_trial(self, trial: Trial) -> None:
        config_path = self.directory / f"config_{trial.id}"
        if config_path.exists():
            raise TrialAlreadyExistsError(trial.id, config_path)
        trials = self.latest()
        with self.cache_path.open("wb") as f:
            trials[trial.id] = trial
            pickle.dump(trials, f)

        config_path.mkdir(parents=True, exist_ok=True)
        TrialReaderWriter.write(trial, self.directory / f"config_{trial.id}", hints=None)

    def update_trial(
        self,
        trial: Trial,
        *,
        hints: list[TrialWriteHint] | TrialWriteHint | None = None,
    ) -> None:
        trials = self.latest()
        with self.cache_path.open("wb") as f:
            trials[trial.id] = trial
            pickle.dump(trials, f)

        TrialReaderWriter.write(trial, self.directory / f"config_{trial.id}", hints=hints)

    def load_trial_from_disk(self, trial_id: str) -> Trial:
        config_path = self.directory / f"config_{trial_id}"
        if not config_path.exists():
            raise TrialNotFoundError(trial_id, config_path)

        return TrialReaderWriter.read(config_path)


@dataclass
class VersionedResource(Generic[T]):
    resource: T
    path: Path
    read: Callable[[Path], T]
    write: Callable[[T, Path], None]
    version_file: Path
    version: Version = "__not_yet_written__"

    def latest(self) -> T:
        if not self.version_file.exists():
            return self.resource

        file_version = self.version_file.read_text()
        if self.version == file_version:
            return self.resource

        self.resource = self.read(self.path)
        self.version = file_version
        return self.resource

    def update(self, new_resource: T) -> Version:
        self.resource = new_resource
        self.version = make_sha()
        self.version_file.write_text(self.version)
        self.write(new_resource, self.path)
        return self.version

    @classmethod
    def new(
        cls,
        resource: T,
        path: Path,
        read: Callable[[Path], T],
        write: Callable[[T, Path], None],
        version_file: Path,
    ) -> VersionedResource[T]:
        if version_file.exists():
            raise FileExistsError(f"Version file already exists at '{version_file}'.")

        write(resource, path)
        version = make_sha()
        version_file.write_text(version)
        return cls(
            resource=resource,
            path=path,
            read=read,
            write=write,
            version_file=version_file,
            version=version,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        read: Callable[[Path], T],
        write: Callable[[T, Path], None],
        version_file: Path,
    ) -> VersionedResource[T]:
        if not path.exists():
            raise FileNotFoundError(f"Resource not found at '{path}'.")

        return cls(
            resource=read(path),
            path=path,
            read=read,
            write=write,
            version_file=version_file,
            version=version_file.read_text(),
        )


@dataclass
class NePSState:
    """The main state object that holds all the shared state objects."""

    path: Path

    _trial_lock: FileLocker = field(repr=False)
    _trials: TrialRepo = field(repr=False)

    _optimizer_lock: FileLocker = field(repr=False)
    _optimizer_info: VersionedResource[OptimizerInfo] = field(repr=False)
    _seed_snapshot: VersionedResource[SeedSnapshot] = field(repr=False)
    _optimizer_state: VersionedResource[OptimizationState] = field(repr=False)

    _err_lock: FileLocker = field(repr=False)
    _shared_errors: VersionedResource[ErrDump] = field(repr=False)

    def lock_and_read_trials(self) -> dict[str, Trial]:
        """Acquire the state lock and read the trials."""
        with self._trial_lock.lock():
            return self._trials.latest()

    def lock_and_sample_trial(self, optimizer: BaseOptimizer, *, worker_id: str) -> Trial:
        """Acquire the state lock and sample a trial."""
        with self._optimizer_lock.lock():
            with self._trial_lock.lock():
                trials = self._trials.latest()

            trial = self._sample_trial(optimizer, trials=trials, worker_id=worker_id)

            with self._trial_lock.lock():
                self._trials.new_trial(trial)

            return trial

    def lock_and_report_trial_evaluation(
        self,
        trial: Trial,
        report: Report,
        *,
        worker_id: str,
    ) -> None:
        """Acquire the state lock and report the trial evaluation."""
        with self._trial_lock.lock(), self._err_lock.lock():
            self._report_trial_evaluation(trial, report, worker_id=worker_id)

    def _sample_trial(
        self,
        optimizer: BaseOptimizer,
        *,
        worker_id: str,
        trials: dict[str, Trial],
        _sample_hooks: list[Callable] | None = None,
    ) -> Trial:
        """Sample a new trial from the optimizer.

        !!! warning

            Responsibility of locking is on caller.

        Args:
            optimizer: The optimizer to sample the trial from.
            worker_id: The worker that is sampling the trial.
            trials: The current trials.
            _sample_hooks: A list of hooks to apply to the optimizer before sampling.

        Returns:
            The new trial.
        """
        seed_state = self._seed_snapshot.latest()
        opt_state = self._optimizer_state.latest()

        seed_state.set_as_global_seed_state()

        # TODO: Not sure if any existing pre_load hooks required
        # it to be done after `load_results`... I hope not.
        if _sample_hooks is not None:
            for hook in _sample_hooks:
                optimizer = hook(optimizer)

        # NOTE: Re-work this, as the part's that are recomputed
        # do not need to be serialized
        budget = opt_state.budget
        if budget is not None:
            budget = budget.clone()

            # NOTE: All other values of budget are ones that should remain
            # constant, there are currently only these two which are dynamic as
            # optimization unfold
            budget.used_cost_budget = sum(
                trial.report.cost
                for trial in trials.values()
                if trial.report is not None and trial.report.cost is not None
            )
            budget.used_evaluations = len(trials)

        sampled_config_maybe_new_opt_state = optimizer.ask(
            trials=trials,
            budget_info=budget,
        )

        if isinstance(sampled_config_maybe_new_opt_state, tuple):
            sampled_config, new_opt_state = sampled_config_maybe_new_opt_state
        else:
            sampled_config = sampled_config_maybe_new_opt_state
            new_opt_state = opt_state.shared_state

        if sampled_config.previous_config_id is not None:
            previous_trial = trials.get(sampled_config.previous_config_id)
            if previous_trial is None:
                raise ValueError(
                    f"Previous trial '{sampled_config.previous_config_id}' not found."
                )
            previous_trial_location = previous_trial.metadata.location
        else:
            previous_trial_location = None

        trial = Trial.new(
            trial_id=sampled_config.id,
            location="",  # HACK: This will be set by the `TrialRepo` in `put_new`
            config=sampled_config.config,
            previous_trial=sampled_config.previous_config_id,
            previous_trial_location=previous_trial_location,
            time_sampled=time.time(),
            worker_id=worker_id,
        )
        seed_state.recapture()
        self._seed_snapshot.update(seed_state)
        self._optimizer_state.update(
            OptimizationState(budget=opt_state.budget, shared_state=new_opt_state)
        )

        return trial

    def _report_trial_evaluation(
        self,
        trial: Trial,
        report: Report,
        *,
        worker_id: str,
    ) -> None:
        """Update the trial with the evaluation report and update the optimizer state
        accordingly.

        Args:
            trial: The trial that was evaluated.
            report: The evaluation report.
            optimizer: The optimizer to update and get the state from
            worker_id: The worker that evaluated the trial.
        """
        # IMPORTANT: We need to attach the report to the trial before updating the things.
        trial.report = report
        self._trials.update_trial(trial, hints=["report", "metadata", "state"])

        if report.err is not None:
            with self._err_lock.lock():
                err_dump = self._shared_errors.latest()
                err_dump.errs.append(
                    ErrDump.SerializableTrialError(
                        trial_id=trial.id,
                        worker_id=worker_id,
                        err_type=type(report.err).__name__,
                        err=str(report.err),
                        tb=report.tb,
                    )
                )
                self._shared_errors.update(err_dump)

    def all_trial_ids(self) -> list[str]:
        """Get all the trial ids."""
        return self._trials.list_trial_ids()

    def lock_and_get_errors(self) -> ErrDump:
        """Get all the errors that have occurred during the optimization."""
        with self._err_lock.lock():
            return self._shared_errors.latest()

    def lock_and_get_optimizer_info(self) -> OptimizerInfo:
        """Get the optimizer information."""
        with self._optimizer_lock.lock():
            return self._optimizer_info.latest()

    def lock_and_get_optimizer_state(self) -> OptimizationState:
        """Get the optimizer state."""
        with self._optimizer_lock.lock():
            return self._optimizer_state.latest()

    def lock_and_get_trial_by_id(self, trial_id: str) -> Trial:
        """Get a trial by its id."""
        with self._trial_lock.lock():
            return self._trials.load_trial_from_disk(trial_id)

    def unsafe_retry_get_trial_by_id(self, trial_id: str) -> Trial:
        """Get a trial by id but use unsafe retries."""
        for _ in range(N_UNSAFE_RETRIES):
            try:
                return self._trials.load_trial_from_disk(trial_id)
            except TrialNotFoundError as e:
                raise e
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to get trial '%s' due to an error: %s", trial_id, e
                )
                time.sleep(0.1)
                continue

        raise NePSError(
            f"Failed to get trial '{trial_id}' after {N_UNSAFE_RETRIES} retries."
        )

    def put_updated_trial(
        self,
        trial: Trial,
        *,
        hints: list[TrialWriteHint] | TrialWriteHint | None = None,
    ) -> None:
        """Update the trial.

        Args:
            trial: The trial to update.
            hints: The hints to use when updating the trial. Defines what files need
                to be updated.
                If you don't know, leave `None`, this is a micro-optimization.
        """
        with self._trial_lock.lock():
            self._trials.update_trial(trial, hints=hints)

    @overload
    def lock_and_get_next_pending_trial(self) -> Trial | None: ...

    @overload
    def lock_and_get_next_pending_trial(self, n: int | None = None) -> list[Trial]: ...

    def lock_and_get_next_pending_trial(
        self,
        n: int | None = None,
    ) -> Trial | list[Trial] | None:
        """Get the next pending trial."""
        with self._trial_lock.lock():
            trials = self._trials.latest()
            pendings = sorted(
                [
                    trial
                    for trial in trials.values()
                    if trial.state == Trial.State.PENDING
                ],
                key=lambda t: t.metadata.time_sampled,
            )
            if n is None:
                return pendings[0] if pendings else None
            return pendings[:n]

    @classmethod
    def create_or_load(
        cls,
        path: Path,
        *,
        load_only: bool = False,
        optimizer_info: OptimizerInfo | None = None,
        optimizer_state: OptimizationState | None = None,
    ) -> NePSState:
        """Create a new NePSState in a directory or load the existing one
        if it already exists, depending on the argument.

        !!! warning

            We check that the optimizer info in the NePSState on disk matches
            the one that is passed. However we do not lock this check so it
            is possible that if two processes try to create a NePSState at the
            same time, both with different optimizer infos, that one will fail
            to create the NePSState. This is a limitation of the current design.

            In principal, we could allow multiple optimizers to be run and share
            the same set of trials.

        Args:
            path: The directory to create the state in.
            load_only: If True, only load the state and do not create a new one.
            optimizer_info: The optimizer info to use.
            optimizer_state: The optimizer state to use.

        Returns:
            The NePSState.

        Raises:
            NePSError: If the optimizer info on disk does not match the one provided.
        """
        is_new = not path.exists()
        if load_only:
            if is_new:
                raise FileNotFoundError(f"No NePSState found at '{path}'.")
        else:
            assert optimizer_info is not None
            assert optimizer_state is not None

        path.mkdir(parents=True, exist_ok=True)
        config_dir = path / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = path / ".seed_state"
        seed_dir.mkdir(parents=True, exist_ok=True)
        error_dir = path / ".errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        optimizer_state_dir = path / ".optimizer_state"
        optimizer_state_dir.mkdir(parents=True, exist_ok=True)
        optimizer_info_dir = path / ".optimizer_info"
        optimizer_info_dir.mkdir(parents=True, exist_ok=True)

        # We have to do one bit of sanity checking to ensure that the optimzier
        # info on disk manages the one we have recieved, otherwise we are unsure which
        # optimizer is being used.
        # NOTE: We assume that we do not have to worry about a race condition
        # here where we have two different NePSState objects with two different optimizer
        # infos trying to be created at the same time. This avoids the need to lock to
        # check the optimizer info. If this assumption changes, then we would have
        # to first lock before we do this check
        if not is_new:
            _optimizer_info = VersionedResource.load(
                optimizer_info_dir,
                read=ReaderWriterOptimizerInfo.read,
                write=ReaderWriterOptimizerInfo.write,
                version_file=optimizer_info_dir / ".version",
            )
            _optimizer_state = VersionedResource.load(
                optimizer_state_dir,
                read=ReaderWriterOptimizationState.read,
                write=ReaderWriterOptimizationState.write,
                version_file=optimizer_state_dir / ".version",
            )
            _seed_snapshot = VersionedResource.load(
                seed_dir,
                read=ReaderWriterSeedSnapshot.read,
                write=ReaderWriterSeedSnapshot.write,
                version_file=seed_dir / ".version",
            )
            _shared_errors = VersionedResource.load(
                error_dir,
                read=ReaderWriterErrDump.read,
                write=ReaderWriterErrDump.write,
                version_file=error_dir / ".version",
            )
            existing_info = _optimizer_info.latest()
            if not load_only and existing_info != optimizer_info:
                raise NePSError(
                    "The optimizer info on disk does not match the one provided."
                    f"\nOn disk: {existing_info}\nProvided: {optimizer_info}"
                    f"\n\nLoaded the one on disk from {optimizer_info_dir}."
                )
        else:
            assert optimizer_info is not None
            assert optimizer_state is not None
            _optimizer_info = VersionedResource.new(
                resource=optimizer_info,
                path=optimizer_info_dir,
                read=ReaderWriterOptimizerInfo.read,
                write=ReaderWriterOptimizerInfo.write,
                version_file=optimizer_info_dir / ".version",
            )
            _optimizer_state = VersionedResource.new(
                resource=optimizer_state,
                path=optimizer_state_dir,
                read=ReaderWriterOptimizationState.read,
                write=ReaderWriterOptimizationState.write,
                version_file=optimizer_state_dir / ".version",
            )
            _seed_snapshot = VersionedResource.new(
                resource=SeedSnapshot.new_capture(),
                path=seed_dir,
                read=ReaderWriterSeedSnapshot.read,
                write=ReaderWriterSeedSnapshot.write,
                version_file=seed_dir / ".version",
            )
            _shared_errors = VersionedResource.new(
                resource=ErrDump(),
                path=error_dir,
                read=ReaderWriterErrDump.read,
                write=ReaderWriterErrDump.write,
                version_file=error_dir / ".version",
            )

        return cls(
            path=path,
            _trials=TrialRepo(config_dir),
            # Locks,
            _trial_lock=FileLocker(
                lock_path=path / ".configs.lock",
                poll=TRIAL_FILELOCK_POLL,
                timeout=TRIAL_FILELOCK_TIMEOUT,
            ),
            _optimizer_lock=FileLocker(
                lock_path=path / ".state.lock",
                poll=STATE_FILELOCK_POLL,
                timeout=STATE_FILELOCK_TIMEOUT,
            ),
            _err_lock=FileLocker(
                lock_path=error_dir / "errors.lock",
                poll=TRIAL_FILELOCK_POLL,
                timeout=TRIAL_FILELOCK_TIMEOUT,
            ),
            # State
            _optimizer_info=_optimizer_info,
            _optimizer_state=_optimizer_state,
            _seed_snapshot=_seed_snapshot,
            _shared_errors=_shared_errors,
        )
