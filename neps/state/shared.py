"""Wrap a resource in a [`Shared`][neps.state.shared.Shared] that handles updating
and commiting of worker's shared resources to disk.

It's primary purpose is to ensure that resources are locked between workers
when being mutated/read from and that resources only need to be updated
once they are stale.

All resources should implement the [`Resource`][neps.state.resource.Resource]
protocol and be able to be loaded, updated, and committed to and from a directory.

Usage:

    * I want to create a new shared resource.

    ```python
    resource = MyResource(directory, ...)
    shared_resource = Shared.new(resource)  # <--- Written to disk here
    ```

    * I want to load an existing resource from a path, **with a lock**.

    ```python
    loaded_resource: Shared[MyResource] = Shared.load(MyResource, directory)
    ```

    * I want to load an existing resource from a path, **without a lock**.

    ```python
    loaded_resource: Shared[MyResource] = Shared.load_unsafe(MyResource, directory)
    ```

    * I want to **mutate** the resource.

    ```python
    with shared_resource.acquire() as resource:  # <--- Lock acquired here
        resource.foo = "bar"
    # <--- Commited to disk and then lock released here
    ```

    * I want to know if the resource is **locked**.

    ```python
    shared_resource.is_locked()  # <--- Check if locked
    ```

    * I want to **read** the latest state of the resource.

    ```python
    latest = shared_resource.latest()  # <--- Updated if stale
    ```

    * I want to **read** the current state of the resource.

    ```python
    current = shared_resource.current()  # <--- No update
    ```

    * I want to know if I have the latest state of a resource

    ```python
    shared_resource.is_stale()  # <--- Check last modified timestamp on disk
    ```

    * I want to **read** the latest state of the resource and **lock** it.

    ```python
    with shared_resource.acquire(commit=False) as resource:  # <--- Lock acquired here
        print(resource.foo)
    # <--- Lock released here, no commit done

Warning:
    * If you are mutating resources use `acquire(commit=True)` (default) to ensure
        that other workers that will use this resource will see the most up-to-date
        version. If you do not do so, other workers will disregard these changes and
        consider it stale.

"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Generic, Iterator, TypeVar
from typing_extensions import ParamSpec, Self

import portalocker as pl

from neps.state.locker import FileLocker, Locker
from neps.state.versioned_store import VersionedDirectoryStore, VersionedStore

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB

# Wait time between each successive poll to see if state can be grabbed
DEFAULT_SHARED_POLL: float = 0.1

# Timeout before giving up on trying to grab the state, raising an error
DEFAULT_SHARED_TIMEOUT: float | None = None

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


class ResourceUpdatedWithoutAcquiringError(RuntimeError):
    """The shared resource was updated by another process without acquiring the lock."""


class ResourceExistsError(RuntimeError):
    """The shared resource already exists on disk."""


class ResourceNotFoundError(FileNotFoundError):
    """The shared resource does not exist on disk."""


class ResourceFailedToLoadError(RuntimeError):
    """The shared resource failed to load from disk."""


V = TypeVar("V")
K = TypeVar("K")


@dataclass
class Shared(Generic[T, K]):
    """Wrap a resource in a `Shared` that handles updating and commiting of worker's
    shared resources to disk.
    """

    LockException: ClassVar = pl.exceptions.LockException
    ResourceExistsError: ClassVar = ResourceExistsError
    ResourceUpdatedWithoutAcquiringError: ClassVar = ResourceUpdatedWithoutAcquiringError
    ResourceNotExistsError: ClassVar = ResourceNotFoundError
    ResourceFailedToLoadError: ClassVar = ResourceFailedToLoadError

    _current: T
    _version: str
    _store: VersionedStore[T, K]
    _locker: Locker

    @classmethod
    def load(
        cls,
        *,
        store: VersionedStore[T, K],
        locker: Locker,
    ) -> Self:
        with locker.lock():
            data, version = store.get()

        return cls(data, version, store, locker)

    @classmethod
    def new(cls, data: T, *, store: VersionedStore[T, K], locker: Locker) -> Self:
        with locker.lock():
            if store.current_version() is not None:
                raise cls.ResourceExistsError(
                    f"Resource already already exists at '{store.key()}'"
                    f" with version '{store.current_version()}'"
                )
            version = store.put(data, None)

        return cls(data, version, store, locker)

    def pull_latest(self) -> T:
        with self._locker.lock():
            self.unsafe_pull_latest()
        return self._current

    def current(self) -> T:
        return self._current

    def version(self) -> str:
        return self._version

    def is_stale(self) -> bool:
        """Check if the shared state is stale."""
        return self._store.current_version() != self._version

    def is_locked(self) -> bool:
        """Check if the shared resource is locked."""
        return self._locker.is_locked()

    def commit(self, data: T) -> str:
        """Commit a new version of the shared resource to disk.

        Args:
            data: The new data to commit to disk.
        """
        with self._locker.lock():
            self._version = self._store.put(data, self._version)
            self._current = data
        return self._version

    @contextmanager
    def acquire(self) -> Iterator[tuple[T, Callable[[T], None]]]:
        """Acquire the lock and return the latest version of the shared resource."""
        with self._locker.lock():
            self.unsafe_pull_latest()
            yield self._current, self._commit_unlocked

    @contextmanager
    def lock(self) -> Iterator[tuple[T, Callable[[T], None]]]:
        """Just lock the shared resource."""
        with self._locker.lock():
            yield self._current, self._commit_unlocked

    def components(self) -> tuple[T, str, VersionedStore[T, K], Locker]:
        """Get the underlying components managing the shared resource."""
        return self._current, self._version, self._store, self._locker

    def deepcopy(self) -> Self:
        """Create a deep copy of the shared resource."""
        return deepcopy(self)

    def _commit_unlocked(self, data: T) -> None:
        self._version = self._store.put(data, self._version)

    def unsafe_pull_latest(self) -> None:
        if self._store.current_version() != self._version:
            self._current, self._version = self._store.get()

    @staticmethod
    def using_directory(
        data: T,
        directory: Path,
        *,
        serialize: Callable[[T, Path], None],
        deserialize: Callable[[Path], T],
        lockname: str = ".lock",
        version_filename: str = ".version",
    ) -> Shared[T, Path]:
        return Shared.new(
            data,
            store=VersionedDirectoryStore(
                directory=directory,
                read=deserialize,
                write=serialize,
                version_filename=version_filename,
            ),
            locker=FileLocker(directory / lockname),
        )
