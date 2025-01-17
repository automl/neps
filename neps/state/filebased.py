"""Contains reading and writing of various aspects of NePS."""

from __future__ import annotations

import contextlib
import json
import logging
import pprint
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import portalocker as pl

from neps.env import CONFIG_SERIALIZE_FORMAT, ENV_VARS_USED
from neps.state.err_dump import ErrDump
from neps.state.trial import Trial
from neps.utils.files import deserialize, serialize

logger = logging.getLogger(__name__)
K = TypeVar("K")
T = TypeVar("T")

TrialWriteHint: TypeAlias = Literal["metadata", "report", "config"]


@dataclass
class ReaderWriterTrial:
    """ReaderWriter for Trial objects."""

    # Report and config are kept as yaml since they are most likely to be
    # read
    CONFIG_FILENAME = f"config.{CONFIG_SERIALIZE_FORMAT}"
    REPORT_FILENAME = f"report.{CONFIG_SERIALIZE_FORMAT}"

    # Metadata is put as json as it's more likely to be machine read and
    # is much faster.
    METADATA_FILENAME = "metadata.json"

    PREVIOUS_TRIAL_ID_FILENAME = "previous_trial_id.txt"

    @classmethod
    def read(cls, directory: Path) -> Trial:
        """Read a trial from a directory."""
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        report_path = directory / cls.REPORT_FILENAME

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        metadata["state"] = Trial.State(metadata["state"])

        return Trial(
            config=deserialize(config_path, file_format=CONFIG_SERIALIZE_FORMAT),
            metadata=Trial.MetaData(**metadata),
            report=(
                Trial.Report(
                    **deserialize(report_path, file_format=CONFIG_SERIALIZE_FORMAT),
                )
                if report_path.exists()
                else None
            ),
        )

    @classmethod
    def write(  # noqa: C901, PLR0912
        cls,
        trial: Trial,
        directory: Path,
        *,
        hints: Iterable[TrialWriteHint] | TrialWriteHint | None = None,
    ) -> None:
        """Write a trial to a directory.

        Args:
            trial: The trial to write.
            directory: The directory to write the trial to.
            hints: What to write. If None, write everything.
        """
        config_path = directory / cls.CONFIG_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME

        if isinstance(hints, str):
            match hints:
                case "config":
                    serialize(
                        trial.config,
                        config_path,
                        check_serialized=False,
                        file_format=CONFIG_SERIALIZE_FORMAT,
                    )
                case "metadata":
                    data = asdict(trial.metadata)
                    data["state"] = data["state"].value
                    with metadata_path.open("w") as f:
                        json.dump(data, f)

                    if trial.metadata.previous_trial_id is not None:
                        previous_trial_path = directory / cls.PREVIOUS_TRIAL_ID_FILENAME
                        previous_trial_path.write_text(trial.metadata.previous_trial_id)
                case "report":
                    if trial.report is None:
                        raise ValueError(
                            "Cannot write report 'hint' when report is None."
                        )

                    report_path = directory / cls.REPORT_FILENAME
                    _report = asdict(trial.report)
                    if (err := _report.get("err")) is not None:
                        _report["err"] = str(err)

                    serialize(
                        _report,
                        report_path,
                        check_serialized=False,
                        file_format=CONFIG_SERIALIZE_FORMAT,
                    )
                case _:
                    raise ValueError(f"Invalid hint: {hints}")
        elif isinstance(hints, Iterable):
            for hint in hints:
                cls.write(trial, directory, hints=hint)  # type: ignore
        elif hints is None:
            # We don't know, write everything
            cls.write(trial, directory, hints=["config", "metadata"])

            if trial.report is not None:
                cls.write(trial, directory, hints="report")
        else:
            raise ValueError(f"Invalid hint: {hints}")


@dataclass
class ReaderWriterErrDump:
    """ReaderWriter for shared error lists."""

    @classmethod
    def read(cls, path: Path) -> ErrDump:
        """Read an error dump from a file."""
        if not path.exists():
            return ErrDump([])

        with path.open("r") as f:
            data = [json.loads(line) for line in f]

        return ErrDump([ErrDump.SerializableTrialError(**d) for d in data])

    @classmethod
    def write(cls, err_dump: ErrDump, path: Path) -> None:
        """Write an error dump to a file."""
        with path.open("w") as f:
            lines = [json.dumps(asdict(trial_err)) for trial_err in err_dump.errs]
            f.write("\n".join(lines))


FILELOCK_EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB


@dataclass
class FileLocker:
    """File-based locker using `portalocker`."""

    lock_path: Path
    poll: float
    timeout: float | None

    def __post_init__(self) -> None:
        self.lock_path = self.lock_path.resolve().absolute()
        self._lock = pl.Lock(
            self.lock_path,
            check_interval=self.poll,
            timeout=self.timeout,
            flags=FILELOCK_EXCLUSIVE_NONE_BLOCKING,
        )

    @contextmanager
    def lock(self, *, worker_id: str | None = None) -> Iterator[None]:
        """Lock the file.

        Args:
            worker_id: The id of the worker trying to acquire the lock.

                Used for debug messaging purposes.
        """
        try:
            with self._lock:
                if worker_id is not None:
                    logger.debug(
                        "Worker %s acquired lock on %s at %s",
                        worker_id,
                        self.lock_path,
                        time.time(),
                    )

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
        finally:
            if worker_id is not None:
                with contextlib.suppress(Exception):
                    logger.debug(
                        "Worker %s released lock on %s at %s",
                        worker_id,
                        self.lock_path,
                        time.time(),
                    )
