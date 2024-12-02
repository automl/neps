from __future__ import annotations

import json
import logging
import pprint
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias, TypeVar

import numpy as np
import portalocker as pl

from neps.env import (
    ENV_VARS_USED,
)
from neps.state.err_dump import ErrDump
from neps.state.optimizer import BudgetInfo, OptimizationState, OptimizerInfo
from neps.state.seed_snapshot import SeedSnapshot
from neps.state.trial import Trial
from neps.utils.files import deserialize, serialize

logger = logging.getLogger(__name__)
K = TypeVar("K")
T = TypeVar("T")

TrialWriteHint: TypeAlias = Literal["metadata", "report", "state", "config"]


@dataclass
class ReaderWriterTrial:
    """ReaderWriter for Trial objects."""

    # Report and config are kept as yaml since they are most likely to be
    # read
    CONFIG_FILENAME = "config.yaml"
    REPORT_FILENAME = "report.yaml"

    # Metadata is put as json as it's more likely to be machine read and
    # is much faster.
    METADATA_FILENAME = "metadata.json"

    STATE_FILENAME = "state.txt"
    PREVIOUS_TRIAL_ID_FILENAME = "previous_trial_id.txt"

    @classmethod
    def read(cls, directory: Path) -> Trial:
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        state_path = directory / cls.STATE_FILENAME
        report_path = directory / cls.REPORT_FILENAME

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        return Trial(
            config=deserialize(config_path),
            metadata=Trial.MetaData(**metadata),
            state=Trial.State(state_path.read_text(encoding="utf-8").strip()),
            report=(
                Trial.Report(**deserialize(report_path)) if report_path.exists() else None
            ),
        )

    @classmethod
    def write(
        cls,
        trial: Trial,
        directory: Path,
        *,
        hints: list[TrialWriteHint] | TrialWriteHint | None = None,
    ) -> None:
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        state_path = directory / cls.STATE_FILENAME

        if isinstance(hints, str):
            match hints:
                case "config":
                    serialize(trial.config, config_path)
                case "metadata":
                    with metadata_path.open("w") as f:
                        json.dump(asdict(trial.metadata), f)

                    if trial.metadata.previous_trial_id is not None:
                        previous_trial_path = directory / cls.PREVIOUS_TRIAL_ID_FILENAME
                        previous_trial_path.write_text(trial.metadata.previous_trial_id)
                case "report":
                    if trial.report is None:
                        raise ValueError(
                            "Cannot write report 'hint' when report is None."
                        )

                    report_path = directory / cls.REPORT_FILENAME
                    serialize(asdict(trial.report), report_path)
                case "state":
                    state_path.write_text(trial.state.value, encoding="utf-8")
                case _:
                    raise ValueError(f"Invalid hint: {hints}")
        elif hints is None:
            # We don't know, write everything
            serialize(trial.config, config_path)
            with metadata_path.open("w") as f:
                json.dump(asdict(trial.metadata), f)

            if trial.metadata.previous_trial_id is not None:
                previous_trial_path = directory / cls.PREVIOUS_TRIAL_ID_FILENAME
                previous_trial_path.write_text(trial.metadata.previous_trial_id)

            state_path.write_text(trial.state.value, encoding="utf-8")

            if trial.report is not None:
                report_path = directory / cls.REPORT_FILENAME
                serialize(asdict(trial.report), report_path)
        else:
            for hint in hints:
                cls.write(trial, directory, hints=hint)


TrialReaderWriter: Final = ReaderWriterTrial()


@dataclass
class ReaderWriterSeedSnapshot:
    """ReaderWriter for SeedSnapshot objects."""

    # It seems like they're all uint32 but I can't be sure.
    PY_RNG_STATE_DTYPE: ClassVar = np.int64

    PY_RNG_TUPLE_FILENAME: ClassVar = "py_rng.npy"
    NP_RNG_STATE_FILENAME: ClassVar = "np_rng_state.npy"
    TORCH_RNG_STATE_FILENAME: ClassVar = "torch_rng_state.pt"
    TORCH_CUDA_RNG_STATE_FILENAME: ClassVar = "torch_cuda_rng_state.pt"
    SEED_INFO_FILENAME: ClassVar = "seed_info.json"

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
        with seedinfo_path.open("w") as f:
            json.dump(seed_info, f)

        np_rng_state = snapshot.np_rng[1]
        np_rng_state.tofile(np_rng_path)

        if snapshot.torch_rng is not None:
            import torch

            torch.save(snapshot.torch_rng, torch_rng_path)

        if snapshot.torch_cuda_rng is not None:
            import torch

            torch.save(snapshot.torch_cuda_rng, torch_cuda_rng_path)


@dataclass
class ReaderWriterOptimizerInfo:
    """ReaderWriter for OptimizerInfo objects."""

    INFO_FILENAME: ClassVar = "info.yaml"

    @classmethod
    def read(cls, directory: Path) -> OptimizerInfo:
        info_path = directory / cls.INFO_FILENAME
        return OptimizerInfo(info=deserialize(info_path))

    @classmethod
    def write(cls, optimizer_info: OptimizerInfo, directory: Path) -> None:
        info_path = directory / cls.INFO_FILENAME
        serialize(optimizer_info.info, info_path)


# TODO(eddiebergman): If an optimizer wants to store some hefty state, i.e. a numpy array
# or something, this is horribly inefficient and we would need to adapt OptimizerState to
# handle this.
# TODO(eddiebergman): May also want to consider serializing budget into a seperate entity
@dataclass
class ReaderWriterOptimizationState:
    """ReaderWriter for OptimizationState objects."""

    STATE_FILE_NAME: ClassVar = "state.json"

    @classmethod
    def read(cls, directory: Path) -> OptimizationState:
        state_path = directory / cls.STATE_FILE_NAME
        with state_path.open("r") as f:
            state = json.load(f)

        shared_state = state.get("shared_state") or {}
        budget_info = state.get("budget")
        budget = BudgetInfo(**budget_info) if budget_info is not None else None
        return OptimizationState(shared_state=shared_state, budget=budget)

    @classmethod
    def write(cls, info: OptimizationState, directory: Path) -> None:
        info_path = directory / cls.STATE_FILE_NAME
        with info_path.open("w") as f:
            json.dump(asdict(info), f)


@dataclass
class ReaderWriterErrDump:
    """ReaderWriter for shared error lists."""

    @classmethod
    def read(cls, directory: Path) -> ErrDump:
        errors_path = directory / "errors.jsonl"
        with errors_path.open("r") as f:
            data = [json.loads(line) for line in f]

        return ErrDump([ErrDump.SerializableTrialError(**d) for d in data])

    @classmethod
    def write(cls, err_dump: ErrDump, directory: Path) -> None:
        errors_path = directory / "errors.jsonl"
        with errors_path.open("w") as f:
            lines = [json.dumps(asdict(trial_err)) for trial_err in err_dump.errs]
            f.write("\n".join(lines))


FILELOCK_EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB


@dataclass
class FileLocker:
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

    @contextmanager
    def lock(
        self,
        *,
        fail_if_locked: bool = False,
    ) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
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
