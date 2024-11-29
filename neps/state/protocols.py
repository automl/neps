"""This module defines the protocols used by
[`NePSState`][neps.state.neps_state.NePSState] and
[`Synced`][neps.state.synced.Synced] to ensure atomic operations to the state itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar
from typing_extensions import Self

from neps.exceptions import (
    LockFailedError,
    TrialAlreadyExistsError,
    TrialNotFoundError,
    VersionedResourceAlreadyExistsError,
    VersionedResourceDoesNotExistError,
    VersionedResourceRemovedError,
    VersionMismatchError,
)

if TYPE_CHECKING:
    from neps.state import Trial

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")

# https://github.com/MaT1g3R/option/issues/40
K2 = TypeVar("K2")
T2 = TypeVar("T2")

Loc_contra = TypeVar("Loc_contra", contravariant=True)


class Versioner(Protocol):
    """A versioner that can bump the version of a resource.

    It should have some [`current()`][neps.state.protocols.Versioner.current] method
    to give the current version tag of a resource and a
    [`bump()`][neps.state.protocols.Versioner.bump] method to provide a new version tag.

    These [`current()`][neps.state.protocols.Versioner.current] and
    [`bump()`][neps.state.protocols.Versioner.bump] methods do not need to be atomic
    but they should read/write to external state, i.e. file-system, database, etc.
    """

    def current(self) -> str | None:
        """Return the current version as defined by the external state, i.e.
        the version of the tag on disk.

        Returns:
            The current version if there is one written.
        """
        ...

    def bump(self) -> str:
        """Create a new external version tag.

        Returns:
            The new version tag.
        """
        ...


class Locker(Protocol):
    """A locker that can be used to communicate between workers."""

    LockFailedError: ClassVar = LockFailedError

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Initiate the lock as a context manager, releasing it when done."""
        ...

    def is_locked(self) -> bool:
        """Check if lock is...well, locked.

        Should return True if the resource is locked, even if the lock is held by the
        current worker/process.
        """
        ...


class ReaderWriter(Protocol[T, Loc_contra]):
    """A reader-writer that can read and write some resource T with location Loc.

    For example, a `ReaderWriter[Trial, Path]` indicates a class that can read and write
    trials, given some `Path`.
    """

    CHEAP_LOCKLESS_READ: ClassVar[bool]
    """Whether reading the contents of the resource is cheap, cheap enough to be
    most likely safe without a lock if outdated information is acceptable.

    This is currently used to help debugging instances of a VersionMismatchError
    to see what the current state is and what was attempted to be written.
    """

    def read(self, loc: Loc_contra, /) -> T:
        """Read the resource at the given location."""
        ...

    def write(self, value: T, loc: Loc_contra, /) -> None:
        """Write the resource at the given location."""
        ...


class TrialRepo(Protocol[K]):
    """A repository of trials.

    The primary purpose of this protocol is to ensure consistent access to trial,
    the ability to put in a new trial and know about the trials that are stored there.
    """

    TrialAlreadyExistsError: ClassVar = TrialAlreadyExistsError
    TrialNotFoundError: ClassVar = TrialNotFoundError

    def all_trial_ids(self) -> set[str]:
        """List all the trial ids in this trial Repo."""
        ...

    def get_by_id(self, trial_id: str) -> Synced[Trial, K]:
        """Get a trial by its id."""
        ...

    def put_new(self, trial: Trial) -> Synced[Trial, K]:
        """Put a new trial in the repo."""
        ...

    def all(self) -> dict[str, Synced[Trial, K]]:
        """Get all trials in the repo."""
        ...

    def pending(self) -> Iterable[tuple[str, Trial]]:
        """Get all pending trials in the repo.

        !!! note
            This should return trials in the order in which they should be next evaluated,
            usually the order in which they were put in the repo.
        """
        ...


@dataclass
class VersionedResource(Generic[T, K]):
    """A resource that will be read if it needs to update to the latest version.

    Relies on 3 main components:
    * A [`Versioner`][neps.state.protocols.Versioner] to manage the versioning of the
        resource.
    * A [`ReaderWriter`][neps.state.protocols.ReaderWriter] to read and write the
        resource.
    * The location of the resource that can be used for the reader-writer.
    """

    VersionMismatchError: ClassVar = VersionMismatchError
    VersionedResourceDoesNotExistsError: ClassVar = VersionedResourceDoesNotExistError
    VersionedResourceAlreadyExistsError: ClassVar = VersionedResourceAlreadyExistsError
    VersionedResourceRemovedError: ClassVar = VersionedResourceRemovedError

    _current: T
    _location: K
    _version: str
    _versioner: Versioner
    _reader_writer: ReaderWriter[T, K]

    @staticmethod
    def new(
        *,
        data: T2,
        location: K2,
        versioner: Versioner,
        reader_writer: ReaderWriter[T2, K2],
    ) -> VersionedResource[T2, K2]:
        """Create a new VersionedResource.

        This will create a new resource if it doesn't exist, otherwise,
        if it already exists, it will raise an error.

        Use [`load()`][neps.state.protocols.VersionedResource.load] if you want to
        load an existing resource.

        Args:
            data: The data to be stored.
            location: The location where the data will be stored.
            versioner: The versioner to be used.
            reader_writer: The reader-writer to be used.

        Returns:
            A new VersionedResource

        Raises:
            VersionedResourceAlreadyExistsError: If a versioned resource already exists
                at the given location.
        """
        current_version = versioner.current()
        if current_version is not None:
            raise VersionedResourceAlreadyExistsError(
                f"A versioned resource already already exists at '{location}'"
                f" with version '{current_version}'"
            )

        version = versioner.bump()
        reader_writer.write(data, location)
        return VersionedResource(
            _current=data,
            _location=location,
            _version=version,
            _versioner=versioner,
            _reader_writer=reader_writer,
        )

    @classmethod
    def load(
        cls,
        *,
        location: K2,
        versioner: Versioner,
        reader_writer: ReaderWriter[T2, K2],
    ) -> VersionedResource[T2, K2]:
        """Load an existing VersionedResource.

        This will load an existing resource if it exists, otherwise, it will raise an
        error.

        Use [`new()`][neps.state.protocols.VersionedResource.new] if you want to
        create a new resource.

        Args:
            location: The location of the resource.
            versioner: The versioner to be used.
            reader_writer: The reader-writer to be used.

        Returns:
            A VersionedResource

        Raises:
            VersionedResourceDoesNotExistsError: If no versioned resource exists at
                the given location.
        """
        version = versioner.current()
        if version is None:
            raise cls.VersionedResourceDoesNotExistsError(
                f"No versioned resource exists at '{location}'."
            )
        data = reader_writer.read(location)
        return VersionedResource(
            _current=data,
            _location=location,
            _version=version,
            _versioner=versioner,
            _reader_writer=reader_writer,
        )

    def sync_and_get(self) -> T:
        """Get the data and version of the resource."""
        self.sync()
        return self._current

    def sync(self) -> None:
        """Sync the resource with the latest version."""
        current_version = self._versioner.current()
        if current_version is None:
            raise self.VersionedResourceRemovedError(
                f"Versioned resource at '{self._location}' has been removed!"
                f" Last known version was '{self._version}'."
            )

        if self._version != current_version:
            self._current = self._reader_writer.read(self._location)
            self._version = current_version

    def put(self, data: T) -> None:
        """Put the data and version of the resource.

        Raises:
            VersionMismatchError: If the version of the resource is not the same as the
                current version. This implies that the resource has been updated by
                another worker.
        """
        current_version = self._versioner.current()
        if self._version != current_version:
            # We will attempt to do a lockless read on the contents of the items, as this
            # would allow us to better debug in the error raised below.
            if self._reader_writer.CHEAP_LOCKLESS_READ:
                current_contents = self._reader_writer.read(self._location)
                extra_msg = (
                    f"\nThe attempted write was: {data}\n"
                    f"The current contents are: {current_contents}"
                )
            else:
                extra_msg = ""

            raise self.VersionMismatchError(
                f"Version mismatch - ours: '{self._version}', remote: '{current_version}'"
                f" Tried to put data at '{self._location}'. Doing so would overwrite"
                " changes made by another worker. The solution is to pull the latest"
                " version of the resource and try again."
                " The most possible reasons for this error is that a lock was not"
                " utilized when getting this resource before putting it back."
                f"{extra_msg}"
            )

        self._reader_writer.write(data, self._location)
        self._current = data
        self._version = self._versioner.bump()

    def current(self) -> T:
        """Get the current data of the resource."""
        return self._current

    def is_stale(self) -> bool:
        """Check if the resource is stale."""
        return self._version != self._versioner.current()

    def location(self) -> K:
        """Get the location of the resource."""
        return self._location


@dataclass
class Synced(Generic[T, K]):
    """Manages a versioned resource but it's methods also implement locking procedures
    for accessing it.

    Its types are parametrized by two type variables:

    * `T` is the type of the data stored in the resource.
    * `K` is the type of the location of the resource, for example `Path`

    This wraps a [`VersionedResource`][neps.state.protocols.VersionedResource] and
    additionally provides utility to perform atmoic operations on it using a
    [`Locker`][neps.state.protocols.Locker].

    This is used by [`NePSState`][neps.state.neps_state.NePSState] to manage the state
    of trials and other shared resources.

    It consists of 2 main components:

    * A [`VersionedResource`][neps.state.protocols.VersionedResource] to manage the
        versioning of the resource.
    * A [`Locker`][neps.state.protocols.Locker] to manage the locking of the resource.

    The primary methods to interact with a resource that is behined a `Synced` are:

    * [`synced()`][neps.state.protocols.Synced.synced] to get the data of the resource
        after syncing it to it's latest verison.
    * [`acquire()`][neps.state.protocols.Synced.acquire] context manager to get latest
        version of the data while also mainting a lock on it. This additionally provides
        a `put()` operation to put the data back. This can primarily be used to get the
        data, perform some mutation on it and then put it back, while not allowing other
        workers access to the data.
    """

    LockFailedError: ClassVar = Locker.LockFailedError
    VersionedResourceRemovedError: ClassVar = (
        VersionedResource.VersionedResourceRemovedError
    )
    VersionMismatchError: ClassVar = VersionedResource.VersionMismatchError
    VersionedResourceAlreadyExistsError: ClassVar = (
        VersionedResource.VersionedResourceAlreadyExistsError
    )
    VersionedResourceDoesNotExistsError: ClassVar = (
        VersionedResource.VersionedResourceDoesNotExistsError
    )

    _resource: VersionedResource[T, K]
    _locker: Locker

    @classmethod
    def new(
        cls,
        *,
        locker: Locker,
        data: T2,
        location: K2,
        versioner: Versioner,
        reader_writer: ReaderWriter[T2, K2],
    ) -> Synced[T2, K2]:
        """Create a new Synced resource.

        This will create a new resource if it doesn't exist, otherwise,
        if it already exists, it will raise an error.

        Use [`load()`][neps.state.protocols.Synced.load] if you want to load an existing
        resource. Use [`new_or_load()`][neps.state.protocols.Synced.new_or_load] if you
        want to create a new resource if it doesn't exist, otherwise load an existing
        resource.

        Args:
            locker: The locker to be used.
            data: The data to be stored.
            location: The location where the data will be stored.
            versioner: The versioner to be used.
            reader_writer: The reader-writer to be used.

        Returns:
            A new Synced resource.

        Raises:
            VersionedResourceAlreadyExistsError: If a versioned resource already exists
                at the given location.
        """
        with locker.lock():
            vr = VersionedResource.new(
                data=data,
                location=location,
                versioner=versioner,
                reader_writer=reader_writer,
            )
            return Synced(_resource=vr, _locker=locker)

    @classmethod
    def load(
        cls,
        *,
        locker: Locker,
        location: K2,
        versioner: Versioner,
        reader_writer: ReaderWriter[T2, K2],
    ) -> Synced[T2, K2]:
        """Load an existing Synced resource.

        This will load an existing resource if it exists, otherwise, it will raise an
        error.

        Use [`new()`][neps.state.protocols.Synced.new] if you want to create a new
        resource. Use [`new_or_load()`][neps.state.protocols.Synced.new_or_load] if you
        want to create a new resource if it doesn't exist, otherwise load an existing
        resource.

        Args:
            locker: The locker to be used.
            location: The location of the resource.
            versioner: The versioner to be used.
            reader_writer: The reader-writer to be used.

        Returns:
            A Synced resource.

        Raises:
            VersionedResourceDoesNotExistsError: If no versioned resource exists at
                the given location.
        """
        with locker.lock():
            return Synced(
                _resource=VersionedResource.load(
                    location=location,
                    versioner=versioner,
                    reader_writer=reader_writer,
                ),
                _locker=locker,
            )

    @classmethod
    def new_or_load(
        cls,
        *,
        locker: Locker,
        data: T2,
        location: K2,
        versioner: Versioner,
        reader_writer: ReaderWriter[T2, K2],
    ) -> Synced[T2, K2]:
        """Create a new Synced resource if it doesn't exist, otherwise load it.

        This will create a new resource if it doesn't exist, otherwise, it will load
        an existing resource.

        Use [`new()`][neps.state.protocols.Synced.new] if you want to create a new
        resource and fail otherwise. Use [`load()`][neps.state.protocols.Synced.load]
        if you want to load an existing resource and fail if it doesn't exist.

        Args:
            locker: The locker to be used.
            data: The data to be stored.

                !!! warning

                    This will be ignored if the data already exists.

            location: The location where the data will be stored.
            versioner: The versioner to be used.
            reader_writer: The reader-writer to be used.

        Returns:
            A Synced resource.
        """
        try:
            return Synced.new(
                locker=locker,
                data=data,
                location=location,
                versioner=versioner,
                reader_writer=reader_writer,
            )
        except VersionedResourceAlreadyExistsError:
            return Synced.load(
                locker=locker,
                location=location,
                versioner=versioner,
                reader_writer=reader_writer,
            )

    def synced(self) -> T:
        """Get the data of the resource atomically."""
        with self._locker.lock():
            return self._resource.sync_and_get()

    def location(self) -> K:
        """Get the location of the resource."""
        return self._resource.location()

    def put(self, data: T) -> None:
        """Update the data atomically."""
        with self._locker.lock():
            self._resource.put(data)

    @contextmanager
    def acquire(self) -> Iterator[tuple[T, Callable[[T], None]]]:
        """Acquire the lock and get the data of the resource.

        This is a context manager that returns the data of the resource and a function
        to put the data back.

        !!! note
            This is the primary way to get the resource, mutate it and put it back.
            Otherwise you likely want [`synced()`][neps.state.protocols.Synced.synced]
            or [`put()`][neps.state.protocols.Synced.put].

        Yields:
            A tuple containing the data of the resource and a function to put the data
            back.
        """
        with self._locker.lock():
            self._resource.sync()
            yield self._resource.current(), self._put_unsafe

    def deepcopy(self) -> Self:
        """Create a deep copy of the shared resource."""
        return deepcopy(self)

    def _components(self) -> tuple[T, K, Versioner, ReaderWriter[T, K], Locker]:
        """Get the components of the shared resource."""
        return (
            self._resource.current(),
            self._resource.location(),
            self._resource._versioner,
            self._resource._reader_writer,
            self._locker,
        )

    def _unsynced(self) -> T:
        """Get the current data of the resource **without** locking and syncing it."""
        return self._resource.current()

    def _is_stale(self) -> bool:
        """Check if the data held currently is not the latest version."""
        return self._resource.is_stale()

    def _is_locked(self) -> bool:
        """Check if the resource is locked."""
        return self._locker.is_locked()

    def _put_unsafe(self, data: T) -> None:
        """Put the data without checking for staleness or acquiring the lock.

        !!! warning
            This should only really be called if you know what you're doing.
        """
        self._resource.put(data)
