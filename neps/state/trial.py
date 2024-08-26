"""A trial is a configuration and it's associated data."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Mapping
from typing_extensions import Self

import numpy as np

from neps.exceptions import NePSError
from neps.utils.types import ERROR, ConfigResult, RawConfig

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace

logger = logging.getLogger(__name__)


class NotReportedYetError(NePSError):
    """Raised when trying to access a report that has not been reported yet."""


class State(Enum):
    """The state of a trial."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    FAILED = "failed"
    CRASHED = "crashed"
    CORRUPTED = "corrupted"
    UNKNOWN = "unknown"


@dataclass
class MetaData:
    """Metadata for a trial."""

    id: str
    location: str
    previous_trial_id: Trial.ID | None
    previous_trial_location: str | None
    sampling_worker_id: str
    time_sampled: float

    evaluating_worker_id: str | None = None
    evaluation_duration: float | None = None

    time_submitted: float | None = None
    time_started: float | None = None
    time_end: float | None = None


@dataclass
class Report:
    """A failed report of the evaluation of a configuration."""

    trial_id: Trial.ID
    loss: float | None
    cost: float | None
    learning_curve: list[float] | None  # TODO: Serializing a large list into yaml sucks!
    extra: Mapping[str, Any]
    err: Exception | None
    tb: str | None
    reported_as: Literal["success", "failed", "crashed"]
    evaluation_duration: float | None

    def __post_init__(self) -> None:
        if isinstance(self.err, str):
            self.err = Exception(self.err)  # type: ignore

    def to_deprecate_result_dict(self) -> dict[str, Any] | ERROR:
        """Return the report as a dictionary."""
        if self.reported_as == "success":
            d = {"loss": self.loss, "cost": self.cost, **self.extra}

            # HACK: Backwards compatibility. Not sure how much this is needed
            # but it should be removed once optimizers stop calling the
            # `get_loss`, `get_cost`, `get_learning_curve` methods of `BaseOptimizer`
            # and just use the `Report` directly.
            if "info_dict" not in d or "learning_curve" not in d["info_dict"]:
                d.setdefault("info_dict", {})["learning_curve"] = self.learning_curve
            return d

        return "error"

    def __eq__(self, value: Any, /) -> bool:
        # HACK : Since it could be probably that one of loss or cost is nan,
        # we need a custom comparator for this object
        # HACK : We also have to skip over the `Err` object since when it's deserialized,
        # we can not recover the original object/type.
        if not isinstance(value, Report):
            return False

        other_items = value.__dict__
        for k, v in self.__dict__.items():
            other_v = other_items[k]

            # HACK: Deserialization of `Err` means we can only compare
            # the string representation of the error.
            if k == "err":
                if str(v) != str(other_v):
                    return False
            elif k in ("loss", "cost"):
                if v is not None and np.isnan(v):
                    if other_v is None or not np.isnan(other_v):
                        return False
                elif v != other_v:
                    return False
            elif v != other_v:
                return False

        return True


@dataclass
class Trial:
    """A trial is a configuration and it's associated data."""

    ID: ClassVar = str
    State: ClassVar = State
    Report: ClassVar = Report
    MetaData: ClassVar = MetaData
    NotReportedYetError: ClassVar = NotReportedYetError

    config: Mapping[str, Any]
    metadata: MetaData
    state: State
    report: Report | None

    @classmethod
    def new(
        cls,
        *,
        trial_id: Trial.ID,
        config: Mapping[str, Any],
        location: str,
        previous_trial: Trial.ID | None,
        previous_trial_location: str | None,
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
                location=location,
                time_sampled=time_sampled,
                previous_trial_id=previous_trial,
                previous_trial_location=previous_trial_location,
                sampling_worker_id=worker_id,
            ),
            report=None,
        )

    @property
    def id(self) -> Trial.ID:
        """Return the id of the trial."""
        return self.metadata.id

    def into_config_result(
        self,
        config_to_search_space: Callable[[RawConfig], SearchSpace],
    ) -> ConfigResult:
        """Convert the trial and report to a `ConfigResult` object."""
        if self.report is None:
            raise self.NotReportedYetError("The trial has not been reported yet.")

        result: dict[str, Any] | ERROR
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

    def set_submitted(self, *, time_submitted: float) -> None:
        """Set the trial as submitted."""
        self.metadata.time_submitted = time_submitted
        self.state = State.SUBMITTED

    def set_evaluating(self, *, time_started: float, worker_id: int | str) -> None:
        """Set the trial as in progress."""
        self.metadata.time_started = time_started
        self.metadata.evaluating_worker_id = str(worker_id)
        self.state = State.EVALUATING

    def set_complete(
        self,
        *,
        report_as: Literal["success", "failed", "crashed"],
        time_end: float,
        loss: float | None,
        cost: float | None,
        learning_curve: list[float] | None,
        err: Exception | None,
        tb: str | None,
        extra: Mapping[str, Any] | None,
        evaluation_duration: float | None,
    ) -> Report:
        """Set the report for the trial."""
        if report_as == "success":
            self.state = State.SUCCESS
        elif report_as == "failed":
            self.state = State.FAILED
        elif report_as == "crashed":
            self.state = State.CRASHED
        else:
            raise ValueError(f"Invalid report_as: '{report_as}'")

        self.metadata.time_end = time_end
        self.metadata.evaluation_duration = evaluation_duration

        extra = {} if extra is None else extra

        loss = float(loss) if loss is not None else None
        cost = float(cost) if cost is not None else None
        if learning_curve is not None:
            learning_curve = [float(v) for v in learning_curve]

        return Report(
            trial_id=self.metadata.id,
            reported_as=report_as,
            evaluation_duration=evaluation_duration,
            loss=loss,
            cost=cost,
            learning_curve=learning_curve,
            extra=extra,
            err=err,
            tb=tb,
        )

    def set_corrupted(self) -> None:
        """Set the trial as corrupted."""
        self.state = State.CORRUPTED

    def reset(self) -> None:
        """Reset the trial to a pending state."""
        self.state = State.PENDING
        self.metadata = MetaData(
            id=self.metadata.id,
            location=self.metadata.location,
            previous_trial_id=self.metadata.previous_trial_id,
            previous_trial_location=self.metadata.previous_trial_location,
            time_sampled=self.metadata.time_sampled,
            sampling_worker_id=self.metadata.sampling_worker_id,
        )


def to_config_result(
    trial: Trial,
    report: Report,
    config_to_search_space: Callable[[RawConfig], SearchSpace],
) -> ConfigResult:
    """Convert the trial and report to a `ConfigResult` object."""
    result: dict[str, Any] | ERROR
    if report.reported_as == "success":
        result = {
            **report.extra,
            "loss": report.loss,
            "cost": report.cost,
        }
    else:
        result = "error"

    return ConfigResult(
        trial.id,
        config=config_to_search_space(trial.config),
        result=result,
        metadata=asdict(trial.metadata),
    )
