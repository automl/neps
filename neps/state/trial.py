"""A trial is a configuration and it's associated data."""

from __future__ import annotations

import datetime
import logging
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
)
from typing_extensions import Self, TypeAlias

from neps.state.report import Report
from neps.state.shared import Shared
from neps.utils.files import deserialize, empty_file, serialize
from neps.utils.types import ConfigResult, RawConfig

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace


logger = logging.getLogger(__name__)

TrialID: TypeAlias = str


class NotReportedYetError(RuntimeError):
    """Raised when trying to access a report that has not been reported yet."""


class DeserializedError(Exception):
    """An exception that was deserialized from a file."""


@dataclass
class _TrialPaths:
    directory: Path

    def __post_init__(self) -> None:
        self.config_file = self.directory / "config.yaml"
        self.metadata_file = self.directory / "metadata.yaml"
        self.report_file = self.directory / "report.yaml"

        self.previous_trial_id_file = self.directory / ".previous_config"
        self.version_file = self.directory / ".version_sha"
        self.state_file = self.directory / ".state"


def _deserialize_from_directory(directory: Path) -> Trial:
    """Deserialize a trial from a directory."""
    paths = _TrialPaths(directory)
    state_str = paths.state_file.read_text()
    state = State(state_str)

    if empty_file(paths.config_file):
        raise FileNotFoundError(f"Config file not found at {paths.config_file}")

    config = deserialize(paths.config_file)
    assert isinstance(config, dict)

    if empty_file(paths.metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {paths.metadata_file}")

    metadata = MetaData(**deserialize(paths.metadata_file))

    report = None
    if not empty_file(paths.report_file):
        report_data = deserialize(paths.report_file)
        _err = report_data.get("err")
        report_data["err"] = DeserializedError(_err) if _err is not None else None
        report = Report(**report_data)

    return Trial(state=state, config=config, report=report, metadata=metadata)


def _serialize_to_directory(trial: Trial, directory: Path) -> None:
    """Serialize a trial to directory."""
    paths = _TrialPaths(directory)
    paths.state_file.write_text(trial.state.value)
    serialize(trial.metadata, paths.metadata_file)
    serialize(trial.config, paths.config_file)
    if trial.report is None and paths.report_file.exists():
        isoformat = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_path = paths.report_file.with_suffix(f".old-{isoformat}.yaml")
        paths.report_file.rename(new_path)
    elif trial.report is not None:
        serialize(trial.report, paths.report_file)

    if trial.metadata.previous_trial_id is not None:
        paths.previous_trial_id_file.write_text(trial.metadata.previous_trial_id)


class State(Enum):
    """The state of a trial."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CRASHED = "crashed"
    CORRUPTED = "corrupted"
    UNKNOWN = "unknown"


@dataclass(kw_only=True)
class MetaData:
    """Metadata for a trial."""

    id: str
    previous_trial_id: TrialID | None
    sampling_worker_id: str
    time_sampled: float
    # Mutable
    evaluating_worker_id: str | None = None
    time_end: float | None = None
    time_submitted: float | None = None
    time_started: float | None = None


@dataclass(kw_only=True)
class Trial:
    """A trial is a configuration and it's associated data.

    The object is considered mutable and the global trial currently being
    evaluated can be access using `get_in_progress_trial()`.
    """

    State: ClassVar = State
    Report: ClassVar = Report
    MetaData: ClassVar = MetaData
    NotReportedYetError: ClassVar = NotReportedYetError

    metadata: MetaData
    state: Trial.State
    config: Mapping[str, Any]
    report: Report | None

    @classmethod
    def new(
        cls,
        *,
        trial_id: TrialID,
        config: Mapping[str, Any],
        previous_trial: TrialID | None,
        time_sampled: float,
        worker_id: int | str,
    ) -> Self:
        """Create a new trial object that was just sampled."""
        worker_id = str(worker_id)
        return cls(
            state=State.PENDING,
            config=config,
            metadata=MetaData(
                id=trial_id,
                time_sampled=time_sampled,
                previous_trial_id=previous_trial,
                sampling_worker_id=worker_id,
            ),
            report=None,
        )

    def as_filesystem_shared(self, directory: Path) -> Shared[Trial, Path]:
        """Return the trial as a shared object."""
        return Shared.using_directory(
            self,
            directory,
            serialize=_serialize_to_directory,
            deserialize=_deserialize_from_directory,
            lockname=".trialinfo_lock",
            version_filename=".trialinfo_sha",
        )

    @property
    def id(self) -> TrialID:
        """Return the id of the trial."""
        return self.metadata.id

    def to_config_result(
        self,
        config_to_search_space: Callable[[RawConfig], SearchSpace],
    ) -> ConfigResult:
        """Convert the report to a `ConfigResult` object."""
        if self.report is None:
            raise NotReportedYetError("The trial has not been reported yet.")

        if self.report.reported_as == "success":
            result = {
                **self.report.extra,
                "loss": self.report.loss,
                "cost": self.report.cost,
            }
        else:
            result = "error"

        return ConfigResult(
            self.id,
            config=config_to_search_space(self.config),
            result=result,
            metadata=asdict(self.metadata),
        )

    def set_submitted(self, *, time_submitted: float | None = None) -> None:
        """Set the trial as submitted."""
        self.metadata.time_submitted = (
            time_submitted if time_submitted is not None else time.time()
        )
        self.state = State.SUBMITTED

    def set_in_progress(
        self,
        *,
        time_started: float | None = None,
        worker_id: int | str | None = None,
    ) -> None:
        """Set the trial as in progress."""
        self.metadata.time_started = (
            time_started if time_started is not None else time.time()
        )
        self.metadata.evaluating_worker_id = (
            str(worker_id) if worker_id is not None else str(os.getpid())
        )
        self.state = State.IN_PROGRESS

    def _set_report(
        self,
        *,
        state: Literal[State.SUCCESS, State.FAILED, State.CRASHED],
        time_end: float | None = None,
        loss: float | None = None,
        err: Exception | None = None,
        tb: str | None = None,
        cost: float | None = None,
        extra: Mapping[str, Any] | None = None,
        account_for_cost: bool = False,
    ) -> None:
        """Set the report for the trial."""
        if state == State.SUCCESS:
            self.state = State.SUCCESS
            report_as = "success"
        elif state == State.FAILED:
            self.state = State.FAILED
            report_as = "failed"
        elif state == State.CRASHED:
            self.state = State.CRASHED
            report_as = "crashed"
        else:
            raise ValueError(f"Invalid state {state}")

        self.report = Report(
            trial_id=self.metadata.id,
            reported_as=report_as,
            loss=float(loss) if loss is not None else None,
            cost=float(cost) if cost is not None else None,
            account_for_cost=account_for_cost,
            extra={} if extra is None else extra,
            err=err,
            tb=tb,
        )
        self.metadata.time_end = time.time() if time_end is None else time_end

    def set_success(
        self,
        *,
        time_end: float | None = None,
        loss: float | None = None,
        err: Exception | None = None,
        tb: str | None = None,
        cost: float | None = None,
        extra: Mapping[str, Any] | None = None,
        account_for_cost: bool = False,
    ) -> None:
        """Set the report for the trial."""
        self._set_report(
            state=State.SUCCESS,
            time_end=time_end,
            loss=loss,
            err=err,
            tb=tb,
            cost=cost,
            extra=extra,
            account_for_cost=account_for_cost,
        )

    def set_fail(
        self,
        *,
        time_end: float | None = None,
        loss: float | None = None,
        err: Exception | None = None,
        tb: str | None = None,
        cost: float | None = None,
        extra: Mapping[str, Any] | None = None,
        account_for_cost: bool = False,
    ) -> None:
        """Set the report for the trial."""
        self._set_report(
            state=State.FAILED,
            time_end=time_end,
            loss=loss,
            err=err,
            tb=tb,
            cost=cost,
            extra=extra,
            account_for_cost=account_for_cost,
        )

    def set_crashed(
        self,
        *,
        time_end: float | None = None,
        loss: float | None = None,
        err: Exception | None = None,
        tb: str | None = None,
        cost: float | None = None,
        extra: Mapping[str, Any] | None = None,
        account_for_cost: bool = False,
    ) -> None:
        """Set the report for the trial."""
        self._set_report(
            state=State.CRASHED,
            time_end=time_end,
            loss=loss,
            err=err,
            tb=tb,
            cost=cost,
            extra=extra,
            account_for_cost=account_for_cost,
        )

    def set_corrupted(self) -> None:
        """Set the trial as corrupted."""
        self.state = State.CORRUPTED

    def reset(self) -> None:
        """Reset the trial to a pending state."""
        self.state = State.PENDING
        self.report = None
        self.metadata = MetaData(
            id=self.metadata.id,
            previous_trial_id=self.metadata.previous_trial_id,
            time_sampled=self.metadata.time_sampled,
            sampling_worker_id=self.metadata.sampling_worker_id,
        )
