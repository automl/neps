"""This module houses the implementation of a NePSState that
does everything on the filesystem, i.e. locking, versioning and
storing/loading.

The main components are:
* [`FileVersioner`][neps.state.filebased.FileVersioner]: A versioner that
    stores a version tag on disk, usually for a resource like a Trial.
* [`FileLocker`][neps.state.filebased.FileLocker]: A locker that uses a file
    to lock between processes.
* [`TrialRepoInDirectory`][neps.state.filebased.TrialRepoInDirectory]: A
    repository of Trials that are stored in a directory.
* `ReaderWriterXXX`: Reader/writers for various resources NePSState needs
* [`load_filebased_neps_state`][neps.state.filebased.load_filebased_neps_state]:
    A function to load a NePSState from a directory.
* [`create_filebased_neps_state`][neps.state.filebased.create_filebased_neps_state]:
    A function to create a new NePSState in a directory.
"""

from __future__ import annotations

import json
import logging
import pprint
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Final, TypeVar
from typing_extensions import override
from uuid import uuid4

import numpy as np
import portalocker as pl

from neps.env import (
    ENV_VARS_USED,
    GLOBAL_ERR_FILELOCK_POLL,
    GLOBAL_ERR_FILELOCK_TIMEOUT,
    OPTIMIZER_INFO_FILELOCK_POLL,
    OPTIMIZER_INFO_FILELOCK_TIMEOUT,
    OPTIMIZER_STATE_FILELOCK_POLL,
    OPTIMIZER_STATE_FILELOCK_TIMEOUT,
    SEED_SNAPSHOT_FILELOCK_POLL,
    SEED_SNAPSHOT_FILELOCK_TIMEOUT,
    TRIAL_FILELOCK_POLL,
    TRIAL_FILELOCK_TIMEOUT,
)
from neps.exceptions import NePSError
from neps.state.err_dump import ErrDump
from neps.state.neps_state import NePSState
from neps.state.optimizer import BudgetInfo, OptimizationState, OptimizerInfo
from neps.state.protocols import Locker, ReaderWriter, Synced, TrialRepo, Versioner
from neps.state.seed_snapshot import SeedSnapshot
from neps.state.trial import Trial
from neps.utils.files import deserialize, serialize

logger = logging.getLogger(__name__)
K = TypeVar("K")
T = TypeVar("T")


def make_sha() -> str:
    """Generate a str hex sha."""
    return uuid4().hex


@dataclass
class FileVersioner(Versioner):
    """A versioner that stores a version tag on disk."""

    version_file: Path

    @override
    def current(self) -> str | None:
        if not self.version_file.exists():
            return None
        return self.version_file.read_text()

    @override
    def bump(self) -> str:
        sha = make_sha()
        self.version_file.write_text(sha)
        return sha


@dataclass
class ReaderWriterTrial(ReaderWriter[Trial, Path]):
    """ReaderWriter for Trial objects."""

    CHEAP_LOCKLESS_READ: ClassVar = True

    CONFIG_FILENAME = "config.yaml"
    METADATA_FILENAME = "metadata.yaml"
    STATE_FILENAME = "state.txt"
    REPORT_FILENAME = "report.yaml"
    PREVIOUS_TRIAL_ID_FILENAME = "previous_trial_id.txt"

    @override
    @classmethod
    def read(cls, directory: Path) -> Trial:
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        state_path = directory / cls.STATE_FILENAME
        report_path = directory / cls.REPORT_FILENAME

        return Trial(
            config=deserialize(config_path),
            metadata=Trial.MetaData(**deserialize(metadata_path)),
            state=Trial.State(state_path.read_text(encoding="utf-8").strip()),
            report=(
                Trial.Report(**deserialize(report_path)) if report_path.exists() else None
            ),
        )

    @override
    @classmethod
    def write(cls, trial: Trial, directory: Path) -> None:
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        state_path = directory / cls.STATE_FILENAME

        serialize(trial.config, config_path)
        serialize(asdict(trial.metadata), metadata_path)
        state_path.write_text(trial.state.value, encoding="utf-8")

        if trial.metadata.previous_trial_id is not None:
            previous_trial_path = directory / cls.PREVIOUS_TRIAL_ID_FILENAME
            previous_trial_path.write_text(trial.metadata.previous_trial_id)

        if trial.report is not None:
            report_path = directory / cls.REPORT_FILENAME
            serialize(asdict(trial.report), report_path)


_StaticReaderWriterTrial: Final = ReaderWriterTrial()


@dataclass
class TrialRepoInDirectory(TrialRepo[Path]):
    """A repository of Trials that are stored in a directory."""

    directory: Path
    _cache: dict[str, Synced[Trial, Path]] = field(default_factory=dict)

    @override
    def all_trial_ids(self) -> set[str]:
        """List all the trial ids in this trial Repo."""
        return {
            config_path.name.replace("config_", "")
            for config_path in self.directory.iterdir()
            if config_path.name.startswith("config_") and config_path.is_dir()
        }

    @override
    def get_by_id(
        self,
        trial_id: str,
        *,
        lock_poll: float = TRIAL_FILELOCK_POLL,
        lock_timeout: float | None = TRIAL_FILELOCK_TIMEOUT,
    ) -> Synced[Trial, Path]:
        """Get a Trial by its ID.

        !!! note

            This will **not** explicitly sync the trial and it is up to the caller
            to do so. Most of the time, the caller should be a NePSState
            object which will do that for you. However if the trial is not in the
            cache, then it will be loaded from disk which requires syncing.

        Args:
            trial_id: The ID of the trial to get.
            lock_poll: The poll time for the file lock.
            lock_timeout: The timeout for the file lock.

        Returns:
            The trial with the given ID.
        """
        trial = self._cache.get(trial_id)
        if trial is not None:
            return trial

        config_path = self.directory / f"config_{trial_id}"
        if not config_path.exists():
            raise TrialRepo.TrialNotFoundError(trial_id, config_path)

        trial = Synced.load(
            location=config_path,
            locker=FileLocker(
                lock_path=config_path / ".lock",
                poll=lock_poll,
                timeout=lock_timeout,
            ),
            versioner=FileVersioner(version_file=config_path / ".version"),
            reader_writer=_StaticReaderWriterTrial,
        )
        self._cache[trial_id] = trial
        return trial

    @override
    def put_new(
        self,
        trial: Trial,
        *,
        lock_poll: float = TRIAL_FILELOCK_POLL,
        lock_timeout: float | None = TRIAL_FILELOCK_TIMEOUT,
    ) -> Synced[Trial, Path]:
        """Put a new Trial into the repository.

        Args:
            trial: The trial to put.
            lock_poll: The poll time for the file lock.
            lock_timeout: The timeout for the file lock.

        Returns:
            The synced trial.

        Raises:
            TrialRepo.TrialAlreadyExistsError: If the trial already exists in the
                repository.
        """
        config_path = self.directory.absolute().resolve() / f"config_{trial.metadata.id}"
        if config_path.exists():
            # This shouldn't exist, we load in the trial to see the current state of it
            # to try determine wtf is going on for logging purposes.
            try:
                shared_trial = Synced.load(
                    location=config_path,
                    locker=FileLocker(
                        lock_path=config_path / ".lock",
                        poll=lock_poll,
                        timeout=lock_timeout,
                    ),
                    versioner=FileVersioner(version_file=config_path / ".version"),
                    reader_writer=_StaticReaderWriterTrial,
                )
                already_existing_trial = shared_trial._unsynced()
                extra_msg = (
                    f"The existing trial is the following: {already_existing_trial}"
                )
            except Exception:  # noqa: BLE001
                extra_msg = "Failed to load the existing trial to provide more info."

            raise TrialRepo.TrialAlreadyExistsError(
                f"Trial '{trial.metadata.id}' already exists as '{config_path}'."
                f" Tried to put in the trial: {trial}."
                f"\n{extra_msg}"
            )

        # HACK: We do this here as there is no way to know where a Trial will
        # be located when it's created...
        trial.metadata.location = str(config_path)
        shared_trial = Synced.new(
            data=trial,
            location=config_path,
            locker=FileLocker(
                lock_path=config_path / ".lock",
                poll=lock_poll,
                timeout=lock_timeout,
            ),
            versioner=FileVersioner(version_file=config_path / ".version"),
            reader_writer=_StaticReaderWriterTrial,
        )
        self._cache[trial.metadata.id] = shared_trial
        return shared_trial

    @override
    def all(self) -> dict[str, Synced[Trial, Path]]:
        """Get a dictionary of all the Trials in the repository.

        !!! note
            See [`get_by_id()`][neps.state.filebased.TrialRepoInDirectory.get_by_id]
            for notes on the trials syncing.
        """
        return {trial_id: self.get_by_id(trial_id) for trial_id in self.all_trial_ids()}

    @override
    def pending(self) -> Iterable[tuple[str, Trial]]:
        pending = [
            (_id, trial, trial.metadata.time_sampled)
            for (_id, t) in self.all().items()
            if (trial := t.synced()).state == Trial.State.PENDING
        ]
        return iter((_id, t) for _id, t, _ in sorted(pending, key=lambda x: x[2]))


@dataclass
class ReaderWriterSeedSnapshot(ReaderWriter[SeedSnapshot, Path]):
    """ReaderWriter for SeedSnapshot objects."""

    CHEAP_LOCKLESS_READ: ClassVar = True

    # It seems like they're all uint32 but I can't be sure.
    PY_RNG_STATE_DTYPE: ClassVar = np.int64

    PY_RNG_TUPLE_FILENAME: ClassVar = "py_rng.npy"
    NP_RNG_STATE_FILENAME: ClassVar = "np_rng_state.npy"
    TORCH_RNG_STATE_FILENAME: ClassVar = "torch_rng_state.pt"
    TORCH_CUDA_RNG_STATE_FILENAME: ClassVar = "torch_cuda_rng_state.pt"
    SEED_INFO_FILENAME: ClassVar = "seed_info.json"

    @override
    @classmethod
    def read(cls, directory: Path) -> SeedSnapshot:
        seedinfo_path = directory / cls.SEED_INFO_FILENAME
        py_rng_path = directory / cls.PY_RNG_TUPLE_FILENAME
        np_rng_path = directory / cls.NP_RNG_STATE_FILENAME
        torch_rng_path = directory / cls.TORCH_RNG_STATE_FILENAME
        torch_cuda_rng_path = directory / cls.TORCH_CUDA_RNG_STATE_FILENAME

        # Load and set pythons rng
        py_rng_state = tuple(
            int(x) for x in np.fromfile(py_rng_path, dtype=cls.PY_RNG_STATE_DTYPE)
        )
        np_rng_state = np.fromfile(np_rng_path, dtype=np.uint32)
        seed_info = deserialize(seedinfo_path)

        torch_rng_path_exists = torch_rng_path.exists()
        torch_cuda_rng_path_exists = torch_cuda_rng_path.exists()

        # By specifying `weights_only=True`, it disables arbitrary object loading
        torch_rng_state = None
        torch_cuda_rng = None
        if torch_rng_path_exists or torch_cuda_rng_path_exists:
            import torch

            if torch_rng_path_exists:
                torch_rng_state = torch.load(torch_rng_path, weights_only=True)

            if torch_cuda_rng_path_exists:
                # By specifying `weights_only=True`, it disables arbitrary object loading
                torch_cuda_rng = torch.load(torch_cuda_rng_path, weights_only=True)

        return SeedSnapshot(
            np_rng=(
                seed_info["np_rng_kind"],
                np_rng_state,
                seed_info["np_pos"],
                seed_info["np_has_gauss"],
                seed_info["np_cached_gauss"],
            ),
            py_rng=(
                seed_info["py_rng_version"],
                py_rng_state,
                seed_info["py_guass_next"],
            ),
            torch_rng=torch_rng_state,
            torch_cuda_rng=torch_cuda_rng,
        )

    @override
    @classmethod
    def write(cls, snapshot: SeedSnapshot, directory: Path) -> None:
        seedinfo_path = directory / cls.SEED_INFO_FILENAME
        py_rng_path = directory / cls.PY_RNG_TUPLE_FILENAME
        np_rng_path = directory / cls.NP_RNG_STATE_FILENAME
        torch_rng_path = directory / cls.TORCH_RNG_STATE_FILENAME
        torch_cuda_rng_path = directory / cls.TORCH_CUDA_RNG_STATE_FILENAME

        py_rng_version, py_rng_state, py_guass_next = snapshot.py_rng

        np.array(py_rng_state, dtype=cls.PY_RNG_STATE_DTYPE).tofile(py_rng_path)

        seed_info = {
            "np_rng_kind": snapshot.np_rng[0],
            "np_pos": snapshot.np_rng[2],
            "np_has_gauss": snapshot.np_rng[3],
            "np_cached_gauss": snapshot.np_rng[4],
            "py_rng_version": py_rng_version,
            "py_guass_next": py_guass_next,
        }
        serialize(seed_info, seedinfo_path)
        np_rng_state = snapshot.np_rng[1]
        np_rng_state.tofile(np_rng_path)

        if snapshot.torch_rng is not None:
            import torch

            torch.save(snapshot.torch_rng, torch_rng_path)

        if snapshot.torch_cuda_rng is not None:
            import torch

            torch.save(snapshot.torch_cuda_rng, torch_cuda_rng_path)


@dataclass
class ReaderWriterOptimizerInfo(ReaderWriter[OptimizerInfo, Path]):
    """ReaderWriter for OptimizerInfo objects."""

    CHEAP_LOCKLESS_READ: ClassVar = True

    INFO_FILENAME: ClassVar = "info.yaml"

    @override
    @classmethod
    def read(cls, directory: Path) -> OptimizerInfo:
        info_path = directory / cls.INFO_FILENAME
        return OptimizerInfo(info=deserialize(info_path))

    @override
    @classmethod
    def write(cls, optimizer_info: OptimizerInfo, directory: Path) -> None:
        info_path = directory / cls.INFO_FILENAME
        serialize(optimizer_info.info, info_path)


# TODO(eddiebergman): If an optimizer wants to store some hefty state, i.e. a numpy array
# or something, this is horribly inefficient and we would need to adapt OptimizerState to
# handle this.
# TODO(eddiebergman): May also want to consider serializing budget into a seperate entity
@dataclass
class ReaderWriterOptimizationState(ReaderWriter[OptimizationState, Path]):
    """ReaderWriter for OptimizationState objects."""

    CHEAP_LOCKLESS_READ: ClassVar = True

    STATE_FILE_NAME: ClassVar = "state.yaml"

    @override
    @classmethod
    def read(cls, directory: Path) -> OptimizationState:
        state_path = directory / cls.STATE_FILE_NAME
        state = deserialize(state_path)
        budget_info = state.get("budget")
        budget = BudgetInfo(**budget_info) if budget_info is not None else None
        return OptimizationState(
            shared_state=state.get("shared_state") or {},
            budget=budget,
        )

    @override
    @classmethod
    def write(cls, info: OptimizationState, directory: Path) -> None:
        info_path = directory / cls.STATE_FILE_NAME
        serialize(asdict(info), info_path)


@dataclass
class ReaderWriterErrDump(ReaderWriter[ErrDump, Path]):
    """ReaderWriter for shared error lists."""

    CHEAP_LOCKLESS_READ: ClassVar = True

    name: str

    @override
    def read(self, directory: Path) -> ErrDump:
        errors_path = directory / f"{self.name}-errors.jsonl"
        with errors_path.open("r") as f:
            data = [json.loads(line) for line in f]

        return ErrDump([ErrDump.SerializableTrialError(**d) for d in data])

    @override
    def write(self, err_dump: ErrDump, directory: Path) -> None:
        errors_path = directory / f"{self.name}-errors.jsonl"
        with errors_path.open("w") as f:
            lines = [json.dumps(asdict(trial_err)) for trial_err in err_dump.errs]
            f.write("\n".join(lines))


FILELOCK_EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB


@dataclass
class FileLocker(Locker):
    """File-based locker using `portalocker`.

    [`FileLocker`][neps.state.locker.file.FileLocker] implements
    the [`Locker`][neps.state.locker.locker.Locker] protocol using
    `portalocker` to lock a file between processes with a shared
    filesystem.
    """

    lock_path: Path
    poll: float
    timeout: float | None

    def __post_init__(self) -> None:
        self.lock_path = self.lock_path.resolve().absolute()

    @override
    def is_locked(self) -> bool:
        if not self.lock_path.exists():
            return False
        try:
            with self.lock(fail_if_locked=True):
                pass
            return False
        except pl.exceptions.LockException:
            return True

    @override
    @contextmanager
    def lock(
        self,
        *,
        fail_if_locked: bool = False,
    ) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        logger.debug("Acquiring lock on %s", self.lock_path)
        try:
            with pl.Lock(
                self.lock_path,
                check_interval=self.poll,
                timeout=self.timeout,
                flags=FILELOCK_EXCLUSIVE_NONE_BLOCKING,
                fail_when_locked=fail_if_locked,
            ):
                yield
        except pl.exceptions.LockException as e:
            raise pl.exceptions.LockException(
                f"Failed to acquire lock after timeout of {self.timeout} seconds."
                " This most likely indicates that another process has crashed while"
                " holding the lock."
                f"\n\nLock path: {self.lock_path}"
                "\n\nIf you belive this is not the case, you can set some of these"
                " environment variables to increase the timeout:"
                f"\n\n{pprint.pformat(ENV_VARS_USED)}"
            ) from e
        logger.debug("Released lock on %s", self.lock_path)


def load_filebased_neps_state(directory: Path) -> NePSState[Path]:
    """Load a NePSState from a directory.

    Args:
        directory: The directory to load the state from.

    Returns:
        The loaded NePSState.

    Raises:
        FileNotFoundError: If no NePSState is found at the given directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"No NePSState found at '{directory}'.")
    directory.mkdir(parents=True, exist_ok=True)
    config_dir = directory / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = directory / ".seed_state"
    seed_dir.mkdir(parents=True, exist_ok=True)
    error_dir = directory / ".errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    optimizer_state_dir = directory / ".optimizer_state"
    optimizer_state_dir.mkdir(parents=True, exist_ok=True)
    optimizer_info_dir = directory / ".optimizer_info"
    optimizer_info_dir.mkdir(parents=True, exist_ok=True)

    return NePSState(
        location=str(directory.absolute().resolve()),
        _trials=TrialRepoInDirectory(config_dir),
        _optimizer_info=Synced.load(
            location=optimizer_info_dir,
            versioner=FileVersioner(version_file=optimizer_info_dir / ".version"),
            locker=FileLocker(
                lock_path=optimizer_info_dir / ".lock",
                poll=OPTIMIZER_INFO_FILELOCK_POLL,
                timeout=OPTIMIZER_INFO_FILELOCK_TIMEOUT,
            ),
            reader_writer=ReaderWriterOptimizerInfo(),
        ),
        _seed_state=Synced.load(
            location=seed_dir,
            reader_writer=ReaderWriterSeedSnapshot(),
            versioner=FileVersioner(version_file=seed_dir / ".version"),
            locker=FileLocker(
                lock_path=seed_dir / ".lock",
                poll=SEED_SNAPSHOT_FILELOCK_POLL,
                timeout=SEED_SNAPSHOT_FILELOCK_TIMEOUT,
            ),
        ),
        _shared_errors=Synced.load(
            location=error_dir,
            reader_writer=ReaderWriterErrDump("all"),
            versioner=FileVersioner(version_file=error_dir / ".all.version"),
            locker=FileLocker(
                lock_path=error_dir / ".all.lock",
                poll=GLOBAL_ERR_FILELOCK_POLL,
                timeout=GLOBAL_ERR_FILELOCK_TIMEOUT,
            ),
        ),
        _optimizer_state=Synced.load(
            location=optimizer_state_dir,
            reader_writer=ReaderWriterOptimizationState(),
            versioner=FileVersioner(version_file=optimizer_state_dir / ".version"),
            locker=FileLocker(
                lock_path=optimizer_state_dir / ".lock",
                poll=OPTIMIZER_STATE_FILELOCK_POLL,
                timeout=OPTIMIZER_STATE_FILELOCK_TIMEOUT,
            ),
        ),
    )


def create_or_load_filebased_neps_state(
    directory: Path,
    *,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> NePSState[Path]:
    """Create a new NePSState in a directory or load the existing one
    if it already exists.

    !!! warning

        We check that the optimizer info in the NePSState on disk matches
        the one that is passed. However we do not lock this check so it
        is possible that if two processes try to create a NePSState at the
        same time, both with different optimizer infos, that one will fail
        to create the NePSState. This is a limitation of the current design.

        In principal, we could allow multiple optimizers to be run and share
        the same set of trials.

    Args:
        directory: The directory to create the state in.
        optimizer_info: The optimizer info to use.
        optimizer_state: The optimizer state to use.

    Returns:
        The NePSState.

    Raises:
        NePSError: If the optimizer info on disk does not match the one provided.
    """
    is_new = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    config_dir = directory / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = directory / ".seed_state"
    seed_dir.mkdir(parents=True, exist_ok=True)
    error_dir = directory / ".errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    optimizer_state_dir = directory / ".optimizer_state"
    optimizer_state_dir.mkdir(parents=True, exist_ok=True)
    optimizer_info_dir = directory / ".optimizer_info"
    optimizer_info_dir.mkdir(parents=True, exist_ok=True)

    # We have to do one bit of sanity checking to ensure that the optimzier
    # info on disk manages the one we have recieved, otherwise we are unsure which
    # optimizer is being used.
    # NOTE: We assume that we do not have to worry about a race condition
    # here where we have two different NePSState objects with two different optimizer
    # infos trying to be created at the same time. This avoids the need to lock to
    # check the optimizer info. If this assumption changes, then we would have
    # to first lock before we do this check
    optimizer_info_reader_writer = ReaderWriterOptimizerInfo()
    if not is_new:
        existing_info = optimizer_info_reader_writer.read(optimizer_info_dir)
        if existing_info != optimizer_info:
            raise NePSError(
                "The optimizer info on disk does not match the one provided."
                f"\nOn disk: {existing_info}\nProvided: {optimizer_info}"
                f"\n\nLoaded the one on disk from {optimizer_info_dir}."
            )

    return NePSState(
        location=str(directory.absolute().resolve()),
        _trials=TrialRepoInDirectory(config_dir),
        _optimizer_info=Synced.new_or_load(
            data=optimizer_info,  # type: ignore
            location=optimizer_info_dir,
            versioner=FileVersioner(version_file=optimizer_info_dir / ".version"),
            locker=FileLocker(
                lock_path=optimizer_info_dir / ".lock",
                poll=OPTIMIZER_INFO_FILELOCK_POLL,
                timeout=OPTIMIZER_INFO_FILELOCK_TIMEOUT,
            ),
            reader_writer=ReaderWriterOptimizerInfo(),
        ),
        _seed_state=Synced.new_or_load(
            data=SeedSnapshot.new_capture(),
            location=seed_dir,
            reader_writer=ReaderWriterSeedSnapshot(),
            versioner=FileVersioner(version_file=seed_dir / ".version"),
            locker=FileLocker(
                lock_path=seed_dir / ".lock",
                poll=SEED_SNAPSHOT_FILELOCK_POLL,
                timeout=SEED_SNAPSHOT_FILELOCK_TIMEOUT,
            ),
        ),
        _shared_errors=Synced.new_or_load(
            data=ErrDump(),
            location=error_dir,
            reader_writer=ReaderWriterErrDump("all"),
            versioner=FileVersioner(version_file=error_dir / ".all.version"),
            locker=FileLocker(
                lock_path=error_dir / ".all.lock",
                poll=GLOBAL_ERR_FILELOCK_POLL,
                timeout=GLOBAL_ERR_FILELOCK_TIMEOUT,
            ),
        ),
        _optimizer_state=Synced.new_or_load(
            data=optimizer_state,
            location=optimizer_state_dir,
            reader_writer=ReaderWriterOptimizationState(),
            versioner=FileVersioner(version_file=optimizer_state_dir / ".version"),
            locker=FileLocker(
                lock_path=optimizer_state_dir / ".lock",
                poll=OPTIMIZER_STATE_FILELOCK_POLL,
                timeout=OPTIMIZER_STATE_FILELOCK_TIMEOUT,
            ),
        ),
    )
