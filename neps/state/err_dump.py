"""Error dump for serializing errors.

This resource is used to store errors that can be serialized and deserialized,
such that they can be shared between workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from neps.exceptions import NePSError


class SerializedError(NePSError):
    """An error the is serialized."""


@dataclass
class SerializableTrialError:
    """Error information for a trial."""

    trial_id: str
    """The ID of the trial."""

    worker_id: str
    """The ID of the worker that evaluated the trial which caused the error."""

    err_type: str
    """The type of the error."""

    err: str
    """The error msg."""

    tb: str | None
    """The traceback of the error."""

    def as_raisable(self) -> SerializedError:
        """Convert the error to a raisable error."""
        return SerializedError(
            f"An error occurred during the evaluation of a trial '{self.trial_id}' which"
            f" was evaluted by worker '{self.worker_id}'. The original error could not"
            " be deserialized but had the following information:"
            "\n\n"
            f"{self.tb}"
            "\n"
            f"{self.err_type}: {self.err}"
        )


@dataclass
class ErrDump:
    """A collection of errors that can be serialized and deserialized."""

    SerializableTrialError: ClassVar = SerializableTrialError

    errs: list[SerializableTrialError] = field(default_factory=list)

    def append(self, err: SerializableTrialError) -> None:
        """Append the an error to the reported errors."""
        return self.errs.append(err)

    def __len__(self) -> int:
        return len(self.errs)

    def __bool__(self) -> bool:
        return bool(self.errs)

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return not self.errs

    def latest_err_as_raisable(self) -> SerializedError | None:
        """Get the latest error."""
        if self.errs:
            return self.errs[-1].as_raisable()
        return None
