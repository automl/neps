"""Exceptions for NePS that don't belong in a specific module."""

from __future__ import annotations


class NePSError(Exception):
    """Base class for all NePS exceptions.

    This allows an easier way to catch all NePS exceptions
    if we inherit all exceptions from this class.
    """


class VersionMismatchError(NePSError):
    """Raised when the version of a resource does not match the expected version."""


class VersionedResourceAlreadyExistsError(NePSError):
    """Raised when a version already exists when trying to create a new versioned
    data.
    """


class VersionedResourceRemovedError(NePSError):
    """Raised when a version already exists when trying to create a new versioned
    data.
    """


class VersionedResourceDoesNotExistError(NePSError):
    """Raised when a versioned resource does not exist at a location."""


class LockFailedError(NePSError):
    """Raised when a lock cannot be acquired."""


class TrialAlreadyExistsError(VersionedResourceAlreadyExistsError):
    """Raised when a trial already exists in the store."""


class TrialNotFoundError(VersionedResourceDoesNotExistError):
    """Raised when a trial already exists in the store."""


class WorkerFailedToGetPendingTrialsError(NePSError):
    """Raised when a worker failed to get pending trials."""


class WorkerRaiseError(NePSError):
    """Raised from a worker when an error is raised.

    Includes additional information on how to recover
    """
