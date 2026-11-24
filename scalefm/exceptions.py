"""Exceptions for ScaleFM that don't belong in a specific module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ScaleFMError(Exception):
    """Base class for all ScaleFM exceptions.

    This allows an easier way to catch all ScaleFM exceptions
    if we inherit all exceptions from this class.
    """


class LockFailedError(ScaleFMError):
    """Raised when a lock cannot be acquired."""


class TrialAlreadyExistsError(ScaleFMError):
    """Raised when a trial already exists in the store."""

    def __init__(self, trial_id: str, *args: Any) -> None:
        """Initialize the exception with the trial id."""
        super().__init__(trial_id, *args)
        self.trial_id = trial_id

    def __str__(self) -> str:
        return f"Trial with id {self.trial_id} already exists!"


class TrialNotFoundError(ScaleFMError):
    """Raised when a trial already exists in the store."""


class WorkerFailedToGetPendingTrialsError(ScaleFMError):
    """Raised when a worker failed to get pending trials."""


class WorkerRaiseError(ScaleFMError):
    """Raised from a worker when an error is raised.

    Includes additional information on how to recover
    """


class TrialValidationError(ValueError):
    """Exception raised when a trial configuration fails validation.

    Attributes:
        config: The configuration dictionary that failed validation.
        message: A detailed error message describing the validation failure.
    """

    def __init__(self, config: Mapping[str, Any], message: str, *args: Any) -> None:
        """Initialize TrialValidationError.

        Args:
            config: The trial configuration that failed validation.
            message: Description of the validation error.
            *args: Additional arguments for the base Exception.
        """
        super().__init__(message, *args)
        self.config = config
        self.message = message

    def __str__(self) -> str:
        return (
            f"Trial validation failed for configuration {self.config}. "
            f"Reason: {self.message}"
        )


class ConstraintViolationError(
    ScaleFMError
):  # TODO: add constraint information to the error message
    """Raised when a configuration violates the constraints of the search space."""


class BudgetExhaustedError(ScaleFMError):
    """Raised by an optimizer when the cost budget left is too small to sample a useful
    configuration. The worker stops right away instead of retrying to sample.
    """
