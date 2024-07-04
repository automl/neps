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
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import IO, ClassVar, Generic, Iterator, TypeVar
from typing_extensions import ParamSpec, Self

import portalocker as pl

from neps.state.resource import Resource

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Resource)
P = ParamSpec("P")

EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB

# Wait time between each successive poll to see if state can be grabbed
DEFAULT_SHARED_POLL: float = 0.1

# Timeout before giving up on trying to grab the state, raising an error
DEFAULT_SHARED_TIMEOUT: float | None = None

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


@contextmanager
def _lock(
    path: Path,
    *,
    poll: float,
    timeout: float | None,
    fail_if_locked: bool,
) -> Iterator[IO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with pl.Lock(
        path,
        check_interval=poll,
        timeout=timeout,
        flags=EXCLUSIVE_NONE_BLOCKING,
        fail_when_locked=fail_if_locked,
    ) as fh:
        yield fh  # We almost never use it but nothing better to yield


class ResourceUpdatedWithoutAcquiringError(RuntimeError):
    """The shared resource was updated by another process without acquiring the lock."""


class ResourceExistsError(RuntimeError):
    """The shared resource already exists on disk."""


class ResourceNotFoundError(FileNotFoundError):
    """The shared resource does not exist on disk."""


class ResourceFailedToLoadError(RuntimeError):
    """The shared resource failed to load from disk."""


@dataclass
class Shared(Generic[T]):
    """Wrap a resource in a `Shared` that handles updating and commiting of worker's
    shared resources to disk.
    """

    LockException: ClassVar = pl.exceptions.LockException
    ResourceExistsError: ClassVar = ResourceExistsError
    ResourceUpdatedWithoutAcquiringError: ClassVar = ResourceUpdatedWithoutAcquiringError
    ResourceNotExistsError: ClassVar = ResourceNotFoundError
    ResourceFailedToLoadError: ClassVar = ResourceFailedToLoadError

    _resource: T

    # NOTE(eddiebergman): Previously I tried to use Path.stat().st_mtime_ns for checking
    # who has the latest version of the resource. Apparently this is not actually at a ns
    # resolution and the exact resolution varies across platforms. To be independant of
    # this we will use a random 16 byte string to represent the last `commit()` of the
    # resource.
    _current_sha: str
    _directory: Path = field(init=False)
    _lock_path: Path = field(init=False)
    _random: Random = field(init=False)
    _has_lock: bool = False

    def __post_init__(self) -> None:
        self._directory = self._resource.directory
        self._lock_path = self._directory / ".lock"
        self._sha_path = self._directory / ".sha"

        # NOTE: Explicitly create a new random object so we do not deal
        # with processes having the same seed from neps sharing seed state.
        self._random = Random()  # noqa: S311

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        kls: type[T],
        poll: float = DEFAULT_SHARED_POLL,
        timeout: float | None = DEFAULT_SHARED_TIMEOUT,
        fail_if_locked: bool = False,
    ) -> Self:
        """Load a shared resource from disk."""
        _lock_path = directory / ".lock"
        _sha_path = directory / ".sha"
        if not _sha_path.exists() or not _lock_path.exists():
            raise ResourceNotFoundError(f"No existing resource found at {directory}.")

        with _lock(_lock_path, poll=poll, timeout=timeout, fail_if_locked=fail_if_locked):
            try:
                item = kls.load(directory)
                shared = cls(item, "")
                shared._current_sha = shared._read_latest_sha()
                return shared
            except Exception as e:
                raise ResourceFailedToLoadError(
                    f"Failed to load resource of type '{kls}' from {directory}."
                ) from e

    @classmethod
    def load_unsafe(cls, directory: Path, *, kls: type[T]) -> Shared[T]:
        """Load a shared resource from disk."""
        _lock_path = directory / ".lock"
        _sha_path = directory / ".sha"
        if not _sha_path.exists() or not _lock_path.exists():
            raise ResourceNotFoundError(f"No existing resource found at {directory}.")

        try:
            item = kls.load(directory)
            shared = cls(item, "")
            shared._current_sha = shared._read_latest_sha()
            return shared
        except Exception as e:
            raise ResourceFailedToLoadError(
                f"Failed to load resource of type '{kls}' from {directory}."
            ) from e

    @classmethod
    def new(cls, resource: T) -> Shared[T]:
        """Create a new shared resource on disk."""
        _lock_path = resource.directory / ".lock"
        _sha_path = resource.directory / ".sha"
        if _sha_path.exists() or _lock_path.exists():
            raise cls.ResourceExistsError(
                f"Resource already exists on disk at {resource.directory}."
            )

        # A new resource should not be locked
        with _lock(
            _lock_path,
            poll=DEFAULT_SHARED_POLL,
            timeout=DEFAULT_SHARED_TIMEOUT,
            fail_if_locked=True,
        ):
            resource.commit()
            shared = cls(resource, resource.__class__, "")
            shared._write_new_sha()
            return shared

    def _read_latest_sha(self) -> str:
        return self._sha_path.read_text()

    def _write_new_sha(self) -> None:
        sha = uuid.uuid4().hex
        self._sha_path.write_text(sha)
        self._current_sha = sha

    def latest(
        self,
        *,
        poll: float = DEFAULT_SHARED_POLL,
        timeout: float | None = DEFAULT_SHARED_TIMEOUT,
    ) -> T:
        """Get the latest state of the resource."""
        self.update(poll=poll, timeout=timeout)
        return self.current()

    def update(
        self,
        *,
        poll: float = DEFAULT_SHARED_POLL,
        timeout: float | None = DEFAULT_SHARED_TIMEOUT,
    ) -> None:
        """Update the shared resource if it is stale."""
        with _lock(self._lock_path, poll=poll, timeout=timeout, fail_if_locked=False):
            self._has_lock = True

            latest_signature = self._read_latest_sha()
            if self._current_sha != latest_signature:
                self._resource.update()
                self._current_sha = latest_signature
        self._has_lock = False

    def current(self) -> T:
        """Get the current state of the resource."""
        return self._resource

    def is_stale(self) -> bool:
        """Check if the shared state is stale."""
        return self._current_sha != self._read_latest_sha()

    def is_locked(self) -> bool:
        """Check if the shared resource is locked."""
        try:
            with _lock(self._lock_path, poll=0, timeout=0, fail_if_locked=True):
                pass
            return False
        except self.LockException:
            return True

    def has_lock(self) -> bool:
        """Check if the shared resource has a lock."""
        return self._has_lock

    @contextmanager
    def acquire(
        self,
        *,
        poll: float = DEFAULT_SHARED_POLL,
        timeout: float | None = DEFAULT_SHARED_TIMEOUT,
        commit: bool = True,
    ) -> Iterator[T]:
        """Context manager to work with the latest state of the resource and mutate
        it.

        Args:
            poll: Polling interval in seconds.
            timeout: Timeout in seconds.
            commit: Whether to commit the resource after the context manager exits.
                If disabled, the state of the resource in this process may be updated
                but unknown to any other workers.

                !!! warning "Committing"

                    If you are mutating the resource, you should leave `commit=True`
                    to ensure that other workers that will use this resource
                    will see the most up-to-date version.
        """
        with _lock(self._lock_path, poll=poll, timeout=timeout, fail_if_locked=False):
            self._has_lock = True
            # Update from disk if it's there and stale
            # If it's not stale, we can skip this step as we have the latest
            # version.
            if self._directory.exists():
                latest_ref = self._read_latest_sha()
                if self._current_sha != latest_ref:
                    self._resource.update()
                    self._current_sha = latest_ref

            yield self._resource

            # Safety check that we should keep for a while.
            # Basically if something else
            latest_ref = self._read_latest_sha()
            if self._current_sha != latest_ref:
                raise self.ResourceUpdatedWithoutAcquiringError(
                    "State was modified outside of `acquire()` context."
                    f" Process: {os.getpid()}! This is most likely a bug."
                )

            if commit:
                self._resource.commit()
                self._write_new_sha()

        self._has_lock = False


G = TypeVar("G")


@dataclass
class TestClass(Generic[G]):
    v: int
    t: type[G]

    @classmethod
    def do(cls, v: int, t: type[G]) -> Self:
        return cls(v, t)


x = TestClass(1, str)
