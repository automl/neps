"""A trial is a configuration and it's associated data."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal
from typing_extensions import Self

import numpy as np

from neps.exceptions import NePSError

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
    state: State
    previous_trial_id: str | None
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

    trial_id: str
    objective_to_minimize: float | None
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

    def __eq__(self, value: Any, /) -> bool:
        # HACK : Since it could be probably that one of objective_to_minimize or cost is
        # nan, we need a custom comparator for this object
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
            elif k in ("objective_to_minimize", "cost"):
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

    State: ClassVar = State
    Report: ClassVar = Report
    MetaData: ClassVar = MetaData
    NotReportedYetError: ClassVar = NotReportedYetError

    config: Mapping[str, Any]
    metadata: MetaData
    report: Report | None

    @classmethod
    def new(
        cls,
        *,
        trial_id: str,
        config: Mapping[str, Any],
        location: str,
        previous_trial: str | None,
        previous_trial_location: str | None,
        time_sampled: float,
        worker_id: int | str,
    ) -> Self:
        """Create a new trial object that was just sampled."""
        worker_id = str(worker_id)
        return cls(
            config=config,
            metadata=MetaData(
                id=trial_id,
                state=State.PENDING,
                location=location,
                time_sampled=time_sampled,
                previous_trial_id=previous_trial,
                previous_trial_location=previous_trial_location,
                sampling_worker_id=worker_id,
            ),
            report=None,
        )

    @property
    def id(self) -> str:
        """Return the id of the trial."""
        return self.metadata.id  # type: ignore

    def set_submitted(self, *, time_submitted: float) -> None:
        """Set the trial as submitted."""
        self.metadata.time_submitted = time_submitted
        self.metadata.state = State.SUBMITTED

    def set_evaluating(self, *, time_started: float, worker_id: int | str) -> None:
        """Set the trial as in progress."""
        self.metadata.time_started = time_started
        self.metadata.evaluating_worker_id = str(worker_id)
        self.metadata.state = State.EVALUATING

    def set_complete(
        self,
        *,
        report_as: Literal["success", "failed", "crashed"],
        time_end: float,
        objective_to_minimize: float | None,
        cost: float | None,
        learning_curve: list[float] | None,
        err: Exception | None,
        tb: str | None,
        extra: Mapping[str, Any] | None,
        evaluation_duration: float | None,
    ) -> Report:
        """Set the report for the trial."""
        if report_as == "success":
            self.metadata.state = State.SUCCESS
        elif report_as == "failed":
            self.metadata.state = State.FAILED
        elif report_as == "crashed":
            self.metadata.state = State.CRASHED
        else:
            raise ValueError(f"Invalid report_as: '{report_as}'")

        self.metadata.time_end = time_end
        self.metadata.evaluation_duration = evaluation_duration

        extra = {} if extra is None else extra

        objective_to_minimize = (
            float(objective_to_minimize) if objective_to_minimize is not None else None
        )
        cost = float(cost) if cost is not None else None
        if learning_curve is not None:
            learning_curve = [float(v) for v in learning_curve]

        return Report(
            trial_id=self.metadata.id,
            reported_as=report_as,
            evaluation_duration=evaluation_duration,
            objective_to_minimize=objective_to_minimize,
            cost=cost,
            learning_curve=learning_curve,
            extra=extra,
            err=err,
            tb=tb,
        )

    def set_corrupted(self) -> None:
        """Set the trial as corrupted."""
        self.metadata.state = State.CORRUPTED

    def reset(self) -> None:
        """Reset the trial to a pending state."""
        self.metadata = MetaData(
            id=self.metadata.id,
            state=State.PENDING,
            location=self.metadata.location,
            previous_trial_id=self.metadata.previous_trial_id,
            previous_trial_location=self.metadata.previous_trial_location,
            time_sampled=self.metadata.time_sampled,
            sampling_worker_id=self.metadata.sampling_worker_id,
        )
