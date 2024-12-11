"""Exceptions for NePS that don't belong in a specific module."""

from __future__ import annotations

from typing import Any


class NePSError(Exception):
    """Base class for all NePS exceptions.

    This allows an easier way to catch all NePS exceptions
    if we inherit all exceptions from this class.
    """


class LockFailedError(NePSError):
    """Raised when a lock cannot be acquired."""


class TrialAlreadyExistsError(NePSError):
    """Raised when a trial already exists in the store."""

    def __init__(self, trial_id: str, *args: Any) -> None:
        super().__init__(trial_id, *args)
        self.trial_id = trial_id

    def __str__(self) -> str:
        return f"Trial with id {self.trial_id} already exists!"


class TrialNotFoundError(NePSError):
    """Raised when a trial already exists in the store."""


class WorkerFailedToGetPendingTrialsError(NePSError):
    """Raised when a worker failed to get pending trials."""


class WorkerRaiseError(NePSError):
    """Raised from a worker when an error is raised.

    Includes additional information on how to recover
    """
