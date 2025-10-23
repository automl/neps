"""The main state object that holds all the shared state objects.

This object is used to interact with the shared state objects in a safe atomic
manner, such that each worker can create an identical NePSState and interact with
it without having to worry about locking or out-dated information.

For an actual instantiation of this object, see
[`create_or_load_filebased_neps_state()`][neps.state.neps_state.NePSState.create_or_load].
"""

from __future__ import annotations

import io
import logging
import pickle
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, overload

from neps.env import (
    GLOBAL_ERR_FILELOCK_POLL,
    GLOBAL_ERR_FILELOCK_TIMEOUT,
    STATE_FILELOCK_POLL,
    STATE_FILELOCK_TIMEOUT,
    TRIAL_CACHE_MAX_UPDATES_BEFORE_CONSOLIDATION,
    TRIAL_FILELOCK_POLL,
    TRIAL_FILELOCK_TIMEOUT,
)
from neps.exceptions import NePSError, TrialAlreadyExistsError, TrialNotFoundError
from neps.state.err_dump import ErrDump
from neps.state.filebased import (
    FileLocker,
    ReaderWriterErrDump,
    ReaderWriterTrial,
    TrialWriteHint,
)
from neps.state.optimizer import OptimizationState
from neps.state.trial import Report, Trial
from neps.utils.files import atomic_write, deserialize, serialize

if TYPE_CHECKING:
    from neps.optimizers import OptimizerInfo
    from neps.optimizers.optimizer import AskFunction

from neps.utils.common import gc_disabled

logger = logging.getLogger(__name__)


# TODO: Technically we don't need the same Location type for all shared objects.
Loc = TypeVar("Loc")
T = TypeVar("T")

Version: TypeAlias = str

Resource: TypeAlias = Literal[
    "optimizer_info", "optimizer_state", "seed_state", "errors", "configs"
]


N_UNSAFE_RETRIES = 10

CONFIG_PREFIX_LEN = len("config_")


# TODO: Ergonomics of this class sucks
@dataclass
class TrialRepo:
    """A repository for trials that are stored on disk.

    !!! warning

        This class does not implement locking and it is up to the caller to ensure
        there are no race conflicts.
    """

    CACHE_FILE_NAME = ".trial_cache.pkl"
    UPDATE_CONSOLIDATION_LIMIT = TRIAL_CACHE_MAX_UPDATES_BEFORE_CONSOLIDATION

    directory: Path
    cache_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.directory / self.CACHE_FILE_NAME

    def list_trial_ids(self) -> list[str]:
        """List all the trial ids on disk."""
        return [
            config_path.name[CONFIG_PREFIX_LEN:]
            for config_path in self.directory.iterdir()
            if config_path.name.startswith("config_") and config_path.is_dir()
        ]

    def _read_pkl_and_maybe_consolidate(
        self,
        *,
        consolidate: bool | None = None,
    ) -> dict[str, Trial]:
        with self.cache_path.open("rb") as f:
            _bytes = f.read()

        buffer = io.BytesIO(_bytes)
        trials: dict[str, Trial] = {}
        updates: list[Trial] = []
        while True:
            try:
                datum = pickle.load(buffer)  # noqa: S301

                # If it's a `dict`, this is the whol trials cache
                if isinstance(datum, dict):
                    assert len(trials) == 0, "Multiple caches present."
                    trials = datum

                # If it's a `list`, these are multiple updates
                elif isinstance(datum, list):
                    updates.extend(datum)

                # Otherwise it's a single update
                else:
                    assert isinstance(datum, Trial), "Not a trial."
                    updates.append(datum)
            except EOFError:
                break

        trials.update({trial.id: trial for trial in updates})
        if consolidate is True or (
            len(updates) > self.UPDATE_CONSOLIDATION_LIMIT and consolidate is None
        ):
            logger.debug(
                "Consolidating trial cache with %d trials and %d updates.",
                len(trials),
                len(updates),
            )
            pickle_bytes = pickle.dumps(trials, protocol=pickle.HIGHEST_PROTOCOL)
            with atomic_write(self.cache_path, "wb") as f:
                f.write(pickle_bytes)

        return trials

    def latest(self, *, create_cache_if_missing: bool = True) -> dict[str, Trial]:
        """Get the latest trials from the cache."""
        if not self.cache_path.exists():
            if not create_cache_if_missing:
                return {}
            # If we end up with no cache but there are trials on disk, we need to read in.
            if any(path.name.startswith("config_") for path in self.directory.iterdir()):
                trial_ids = self.list_trial_ids()
                trials = {
                    trial_id: self.load_trial_from_disk(trial_id)
                    for trial_id in trial_ids
                }
                pickle_bytes = pickle.dumps(trials, protocol=pickle.HIGHEST_PROTOCOL)
                with atomic_write(self.cache_path, "wb") as f:
                    f.write(pickle_bytes)

            return {}

        return self._read_pkl_and_maybe_consolidate()

    def store_new_trial(self, trial: Trial | list[Trial]) -> None:
        """Write a new trial to disk.

        Raises:
            TrialAlreadyExistsError: If the trial already exists on disk.
        """
        if isinstance(trial, Trial):
            config_path = self.directory / f"config_{trial.id}"
            if config_path.exists():
                raise TrialAlreadyExistsError(trial.id, config_path)

            bytes_ = pickle.dumps(trial, protocol=pickle.HIGHEST_PROTOCOL)
            with atomic_write(self.cache_path, "ab") as f:
                f.write(bytes_)

            config_path.mkdir(parents=True, exist_ok=True)
            ReaderWriterTrial.write(
                trial,
                self.directory / f"config_{trial.id}",
                hints=["config", "metadata"],
            )
        else:
            for child_trial in trial:
                config_path = self.directory / f"config_{child_trial.id}"
                if config_path.exists():
                    raise TrialAlreadyExistsError(child_trial.id, config_path)
                config_path.mkdir(parents=True, exist_ok=True)

            bytes_ = pickle.dumps(trial, protocol=pickle.HIGHEST_PROTOCOL)
            with atomic_write(self.cache_path, "ab") as f:
                f.write(bytes_)

            for child_trial in trial:
                ReaderWriterTrial.write(
                    child_trial,
                    self.directory / f"config_{child_trial.id}",
                    hints=["config", "metadata"],
                )

    def update_trial(
        self,
        trial: Trial,
        *,
        hints: Iterable[TrialWriteHint] | TrialWriteHint | None = ("report", "metadata"),
    ) -> None:
        """Update a trial on disk.

        Args:
            trial: The trial to update.
            hints: The hints to use when updating the trial. Defines what files need
                to be updated.
                If you don't know, leave `None`, this is a micro-optimization.
        """
        bytes_ = pickle.dumps(trial, protocol=pickle.HIGHEST_PROTOCOL)
        with atomic_write(self.cache_path, "ab") as f:
            f.write(bytes_)

        ReaderWriterTrial.write(trial, self.directory / f"config_{trial.id}", hints=hints)

    def load_trial_from_disk(self, trial_id: str) -> Trial:
        """Load a trial from disk.

        Raises:
            TrialNotFoundError: If the trial is not found on disk.
        """
        config_path = self.directory / f"config_{trial_id}"
        if not config_path.exists():
            raise TrialNotFoundError(
                f"Trial {trial_id} not found at expected path of {config_path}."
            )

        return ReaderWriterTrial.read(config_path)


@dataclass
class NePSState:
    """The main state object that holds all the shared state objects."""

    path: Path

    _trial_lock: FileLocker = field(repr=False)
    _trial_repo: TrialRepo = field(repr=False)

    _optimizer_lock: FileLocker = field(repr=False)

    _optimizer_info_path: Path = field(repr=False)
    _optimizer_info: OptimizerInfo = field(repr=False)

    _optimizer_state_path: Path = field(repr=False)
    _optimizer_state: OptimizationState = field(repr=False)

    _err_lock: FileLocker = field(repr=False)
    _shared_errors_path: Path = field(repr=False)
    _shared_errors: ErrDump = field(repr=False)

    new_score: float = float("inf")
    """Tracking of the new incumbent"""

    all_best_configs: list = field(default_factory=list)
    """Trajectory to the newest incbumbent"""

    def lock_and_set_new_worker_id(self, worker_id: str | None = None) -> str:
        """Acquire the state lock and set a new worker id in the optimizer state.

        Args:
            worker_id: The worker id to set. If `None`, a new worker id will be generated.

        Returns:
            The worker id that was set.

        Raises:
                NePSError: If the worker id already exists.
        """
        with self._optimizer_lock.lock():
            with self._optimizer_state_path.open("rb") as f:
                opt_state: OptimizationState = pickle.load(f)  # noqa: S301
                assert isinstance(opt_state, OptimizationState)
                worker_id = (
                    worker_id
                    if worker_id is not None
                    else _get_worker_name(
                        len(opt_state.worker_ids)
                        if opt_state.worker_ids is not None
                        else 0
                    )
                )
                if opt_state.worker_ids and worker_id in opt_state.worker_ids:
                    raise NePSError(
                        f"Worker id '{worker_id}' already exists, \
                        reserved worker ids: {opt_state.worker_ids}"
                    )
                if opt_state.worker_ids is None:
                    opt_state.worker_ids = []

                opt_state.worker_ids.append(worker_id)
            bytes_ = pickle.dumps(opt_state, protocol=pickle.HIGHEST_PROTOCOL)
            with atomic_write(self._optimizer_state_path, "wb") as f:
                f.write(bytes_)
            return worker_id

    def lock_and_read_trials(self) -> dict[str, Trial]:
        """Acquire the state lock and read the trials."""
        with self._trial_lock.lock():
            return self._trial_repo.latest()

    @overload
    def lock_and_sample_trial(
        self, optimizer: AskFunction, *, worker_id: str, n: None = None
    ) -> Trial: ...
    @overload
    def lock_and_sample_trial(
        self, optimizer: AskFunction, *, worker_id: str, n: int
    ) -> list[Trial]: ...

    def lock_and_sample_trial(
        self, optimizer: AskFunction, *, worker_id: str, n: int | None = None
    ) -> Trial | list[Trial]:
        """Acquire the state lock and sample a trial."""
        with self._optimizer_lock.lock():
            with self._trial_lock.lock():
                trials_ = self._trial_repo.latest()

            trials = self._sample_trial(
                optimizer,
                trials=trials_,
                worker_id=worker_id,
                n=n,
            )

            with self._trial_lock.lock():
                self._trial_repo.store_new_trial(trials)

            return trials

    def lock_and_import_trials(
        self, data: list, *, worker_id: str
    ) -> Trial | list[Trial]:
        """Acquire the state lock and import trials from external data.

        Args:
            data: List of trial dictionaries to import.
            worker_id: The worker ID performing the import.

        Returns:
            Trial or list of Trial objects imported.

        Raises:
            ValueError: If data is not a list or trial import fails.
            NePSError: If storing or reporting trials fails.
        """
        with self._optimizer_lock.lock(), gc_disabled():
            trials = Trial.load_from_dict(data=data, worker_id=worker_id)

            with self._trial_lock.lock():
                self._trial_repo.store_new_trial(trials)
                for trial in trials:
                    assert trial.report is not None
                    self._report_trial_evaluation(
                        trial=trial,
                        report=trial.report,
                        worker_id=worker_id,
                    )
            return trials

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

    @overload
    def _sample_trial(
        self,
        optimizer: AskFunction,
        *,
        worker_id: str,
        trials: dict[str, Trial],
        n: int,
    ) -> list[Trial]: ...

    @overload
    def _sample_trial(
        self,
        optimizer: AskFunction,
        *,
        worker_id: str,
        trials: dict[str, Trial],
        n: None,
    ) -> Trial: ...

    def _sample_trial(
        self,
        optimizer: AskFunction,
        *,
        worker_id: str,
        trials: dict[str, Trial],
        n: int | None,
    ) -> Trial | list[Trial]:
        """Sample a new trial from the optimizer.

        !!! warning

            Responsibility of locking is on caller.

        Args:
            optimizer: The optimizer to sample the trial from.
            worker_id: The worker that is sampling the trial.
            n: The number of trials to sample.
            trials: The current trials.

        Returns:
            The new trial.
        """
        with self._optimizer_state_path.open("rb") as f:
            opt_state: OptimizationState = pickle.load(f)  # noqa: S301

        opt_state.seed_snapshot.set_as_global_seed_state()

        assert callable(optimizer)
        if opt_state.budget is not None:
            # NOTE: All other values of budget are ones that should remain
            # constant, there are currently only these two which are dynamic as
            # optimization unfold
            opt_state.budget.used_cost_budget = sum(
                trial.report.cost
                for trial in trials.values()
                if trial.report is not None and trial.report.cost is not None
            )
            opt_state.budget.used_evaluations = len(trials)

        sampled_configs = optimizer(
            trials=trials,
            budget_info=(
                opt_state.budget.clone() if opt_state.budget is not None else None
            ),
            n=n,
        )

        if not isinstance(sampled_configs, list):
            sampled_configs = [sampled_configs]

        # TODO: Not implemented yet.
        shared_state = opt_state.shared_state

        sampled_trials: list[Trial] = []
        for sampled_config in sampled_configs:
            if sampled_config.previous_config_id is not None:
                previous_trial = trials.get(sampled_config.previous_config_id)
                if previous_trial is None:
                    raise ValueError(
                        f"Previous trial '{sampled_config.previous_config_id}' not found."
                    )
                previous_trial_location = previous_trial.metadata.location
            else:
                previous_trial_location = None

            id_str = sampled_config.id
            config_name = f"{sampled_config.id}"
            parts = id_str.split("_rung_")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                config_id, rung_id = map(int, parts)
                config_name = f"{config_id}_rung_{rung_id}"

            trial = Trial.new(
                trial_id=sampled_config.id,
                location=str(self._trial_repo.directory / f"config_{config_name}"),
                config=sampled_config.config,
                previous_trial=sampled_config.previous_config_id,
                previous_trial_location=previous_trial_location,
                time_sampled=time.time(),
                worker_id=worker_id,
            )
            sampled_trials.append(trial)

        opt_state.shared_state = shared_state
        opt_state.seed_snapshot.recapture()
        with self._optimizer_state_path.open("wb") as f:
            pickle.dump(opt_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        if n is None:
            assert len(sampled_trials) == 1
            return sampled_trials[0]

        return sampled_trials

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
        self._trial_repo.update_trial(trial, hints=["report", "metadata"])

        if report.err is not None:
            with self._err_lock.lock():
                err_dump = ReaderWriterErrDump.read(self._shared_errors_path)
                err_dump.errs.append(
                    ErrDump.SerializableTrialError(
                        trial_id=trial.id,
                        worker_id=worker_id,
                        err_type=type(report.err).__name__,
                        err=str(report.err),
                        tb=report.tb,
                    )
                )
                ReaderWriterErrDump.write(err_dump, self._shared_errors_path)

    def all_trial_ids(self) -> list[str]:
        """Get all the trial ids."""
        return self._trial_repo.list_trial_ids()

    def lock_and_get_errors(self) -> ErrDump:
        """Get all the errors that have occurred during the optimization."""
        with self._err_lock.lock():
            return ReaderWriterErrDump.read(self._shared_errors_path)

    def lock_and_get_optimizer_info(self) -> OptimizerInfo:
        """Get the optimizer information."""
        with self._optimizer_lock.lock():
            return _deserialize_optimizer_info(self._optimizer_info_path)

    def lock_and_get_optimizer_state(self) -> OptimizationState:
        """Get the optimizer state."""
        with self._optimizer_lock.lock():  # noqa: SIM117
            with self._optimizer_state_path.open("rb") as f:
                obj = pickle.load(f)  # noqa: S301
                assert isinstance(obj, OptimizationState)
                return obj

    def lock_and_get_trial_by_id(self, trial_id: str) -> Trial:
        """Get a trial by its id."""
        with self._trial_lock.lock():
            return self._trial_repo.load_trial_from_disk(trial_id)

    def unsafe_retry_get_trial_by_id(self, trial_id: str) -> Trial:
        """Get a trial by id but use unsafe retries."""
        for _ in range(N_UNSAFE_RETRIES):
            try:
                return self._trial_repo.load_trial_from_disk(trial_id)
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
            self._trial_repo.update_trial(trial, hints=hints)

    @overload
    def lock_and_get_next_pending_trial(self) -> Trial | None: ...

    @overload
    def lock_and_get_next_pending_trial(self, n: int) -> list[Trial]: ...

    def lock_and_get_next_pending_trial(
        self,
        n: int | None = None,
    ) -> Trial | list[Trial] | None:
        """Get the next pending trial."""
        with self._trial_lock.lock():
            trials = self._trial_repo.latest()
            pendings = sorted(
                [
                    trial
                    for trial in trials.values()
                    if trial.metadata.state == Trial.State.PENDING
                ],
                key=lambda t: t.metadata.time_sampled,
            )
            if n is None:
                return pendings[0] if pendings else None
            return pendings[:n]

    def lock_and_get_current_evaluating_trials(self) -> list[Trial]:
        """Get the current evaluating trials."""
        with self._trial_lock.lock():
            trials = self._trial_repo.latest()
            return [
                trial
                for trial in trials.values()
                if trial.metadata.state == Trial.State.EVALUATING
            ]

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
        path = path.absolute().resolve()
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

        optimizer_info_path = path / "optimizer_info.yaml"
        optimizer_state_path = path / "optimizer_state.pkl"
        shared_errors_path = path / "shared_errors.jsonl"

        # We have to do one bit of sanity checking to ensure that the optimzier
        # info on disk manages the one we have recieved, otherwise we are unsure which
        # optimizer is being used.
        # NOTE: We assume that we do not have to worry about a race condition
        # here where we have two different NePSState objects with two different optimizer
        # infos trying to be created at the same time. This avoids the need to lock to
        # check the optimizer info. If this assumption changes, then we would have
        # to first lock before we do this check
        if not is_new:
            existing_info = _deserialize_optimizer_info(optimizer_info_path)
            if not load_only and existing_info != optimizer_info:
                raise NePSError(
                    "The optimizer info on disk does not match the one provided."
                    f"\nOn disk: {existing_info}\nProvided: {optimizer_info}"
                    f"\n\nLoaded the one on disk from {path}."
                )
            with optimizer_state_path.open("rb") as f:
                optimizer_state = pickle.load(f)  # noqa: S301

            optimizer_info = existing_info
            error_dump = ReaderWriterErrDump.read(shared_errors_path)
        else:
            assert optimizer_info is not None
            assert optimizer_state is not None

            serialize(optimizer_info, path=optimizer_info_path)
            with optimizer_state_path.open("wb") as f:
                pickle.dump(optimizer_state, f, protocol=pickle.HIGHEST_PROTOCOL)

            error_dump = ErrDump([])

        return NePSState(
            path=path,
            _trial_repo=TrialRepo(config_dir),
            # Locks,
            _trial_lock=FileLocker(
                lock_path=path / ".configs.lock",
                poll=TRIAL_FILELOCK_POLL,
                timeout=TRIAL_FILELOCK_TIMEOUT,
            ),
            _optimizer_lock=FileLocker(
                lock_path=path / ".optimizer.lock",
                poll=STATE_FILELOCK_POLL,
                timeout=STATE_FILELOCK_TIMEOUT,
            ),
            _err_lock=FileLocker(
                lock_path=path / ".errors.lock",
                poll=GLOBAL_ERR_FILELOCK_POLL,
                timeout=GLOBAL_ERR_FILELOCK_TIMEOUT,
            ),
            # State
            _optimizer_info_path=optimizer_info_path,
            _optimizer_info=optimizer_info,
            _optimizer_state_path=optimizer_state_path,
            _optimizer_state=optimizer_state,  # type: ignore
            _shared_errors_path=shared_errors_path,
            _shared_errors=error_dump,
        )


def _deserialize_optimizer_info(path: Path) -> OptimizerInfo:
    from neps.optimizers import OptimizerInfo  # Fighting circular import

    deserialized = deserialize(path)
    if "name" not in deserialized or "info" not in deserialized:
        raise NePSError(
            "Invalid optimizer info deserialized from"
            f" {path}. Did not find"
            " keys 'name' and 'info'."
        )
    name = deserialized["name"]
    info = deserialized["info"]
    if not isinstance(name, str):
        raise NePSError(
            f"Invalid optimizer name '{name}' deserialized from {path}. Expected a `str`."
        )

    if not isinstance(info, dict | None):
        raise NePSError(
            f"Invalid optimizer info '{info}' deserialized from"
            f" {path}. Expected a `dict` or `None`."
        )
    return OptimizerInfo(name=name, info=info or {})


def _get_worker_name(idx: int) -> str:
    return f"worker_{idx}"
