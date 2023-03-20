from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from abc import ABC
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import filelock
from typing_extensions import Self

NEPS_LOCKFILE_DIR = "_neps_lockfiles"

import portalocker as pl


def get_user_temp_dir() -> Path:
    """Get the user's temp directory.

    # https://github.com/ray-project/ray/blob/c1707e0dd99a1aae7a82b3e2ff2a6c0f766b4cf5/python/ray/_private/utils.py#L73
    """
    if "NEPS_TMPDIR" in os.environ:
        path = os.environ["RAY_TMPDIR"]
    elif sys.platform.startswith("linux") and "TMPDIR" in os.environ:
        path = os.environ["TMPDIR"]
    elif sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
        # Ideally we wouldn't need this fallback, but keep it for now for
        # for compatibility
        path = os.path.join(os.sep, "tmp")
    else:
        path = tempfile.gettempdir()

    return Path(path).resolve()


class Locker(ABC):
    """A file lock that can be used as a context manager.

    ```python
    # Creates a lockfile in the same directory as the path
    with Locker.local(path, suffix="lock"):
        with path.open("w") as f:
            f.write("hello world")

    # Creates a lockfile linked to this path but stores the lock in TMPDIR
    with Locker.tmplock(path):
        with path.open("w") as f:
            f.write("hello world")
    ```

    You can control the tmpdir by setting the `NEPS_TMPDIR` environment variable,
    or by setting the `tmpdir` argument.

    For finer grained control over the underling filelock you can use the
    `lock` context manager.

    ```python
    locker = Locker.local(path, suffix="lock")
    with locker.lock(timeout=10, poll=0.1):
        with path.open("w") as f:
    ```

    Please see the [`filelock` documentation][https://py-filelock.readthedocs.io/en/latest/api.html#filelock.FileLock]
    for more information.
    """
    Timeout = filelock.Timeout

    def __init__(self, path: Path, lock: filelock.FileLock):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock

    @contextmanager
    def lock(self, *, force_release: bool = False, **kwargs: Any) -> Iterator[bool]:
        """Acquire the lock and release it when the context is exited.

        If the lock can not be acquired within the timeout period, the context
        block is skipped.

        Args:
            force_release: Whether to force release the lock. Passed
            to [`filelock.FileLock.release`][https://py-filelock.readthedocs.io/en/latest/api.html#filelock.BaseFileLock.release].
            **kwargs: Additional keyword arguments to passed
            to [`filelock.FileLock.acquire`][https://py-filelock.readthedocs.io/en/latest/api.html#filelock.BaseFileLock.acquire]
        """
        self.path.touch()
        try:
            self._lock.acquire(**kwargs)
            yield True
        except filelock.Timeout:
            yield False
        finally:
            self._lock.release(force=force_release)

    def acquire(self, **kwargs: Any) -> bool:
        self.path.touch()
        try:
            self._lock.acquire(**kwargs)
            return True
        except filelock.Timeout:
            return False

    @contextmanager
    def release_on_error(self) -> Iterator[Self]:
        try:
            yield self
        finally:
            self._lock.release()

    def release(self, force: bool = False) -> None:
        self._lock.release(force=force)

    def __enter__(self) -> Self:
        self.path.touch()
        self._lock.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self._lock.release()

    @classmethod
    def local(cls, path: Path | str, *, suffix: str = "lock", **kwargs: Any) -> LocalFileLock:
        """Create a lock file for a path in the same directory.

        Args:
            path: The path to lock.
            suffix: The suffix to use for the lock file.
            kwargs: Additional keyword arguments to pass to the filelock.FileLock
                constructor.
        """
        return LocalFileLock(path=path, suffix=suffix, **kwargs)

    @classmethod
    def tmplock(
        cls, path: Path | str, *, tmpdir: Path | str | None = None, **kwargs: Any
    ):
        """Create a lock file for a path in a temporary directory.

        Args:
            path: The path to lock.
            tmpdir: The directory to store the lock file in. Defaults to the
                user's temp directory.
            kwargs: Additional keyword arguments to pass to the filelock.FileLock
        """
        return TmpFileLock(path=path, tmpdir=tmpdir, **kwargs)

    @classmethod
    def nolock(cls, *_: Any, **__: Any) -> nullcontext[None]:
        """Create a no-op lock."""
        return nullcontext(None)


class LocalFileLock(Locker):
    def __init__(self, path: Path | str, *, suffix: str = "lock", **kwargs: Any):
        """Create a new locker.

        Args:
            path: The path to lock.
            suffix: The suffix to use for the lock file.
            kwargs: Additional keyword arguments to pass to the filelock.FileLock
                constructor.
        """
        path = Path(path).resolve()
        if path.is_dir():
            lock_path = path / f".{suffix}"
        else:
            lock_path = path.parent / f"{path.name}.{suffix}"

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(lock_path, **kwargs)
        super().__init__(path=path, lock=lock)


class TmpFileLock(Locker):
    """A context manager for acquiring access to a path."""

    def __init__(
        self, path: Path | str, *, tmpdir: Path | str | None = None, **kwargs: Any
    ):
        """Create a new locker.

        Args:
            path: The path to lock.
            tmpdir: The directory to store the lock file in. Defaults to the
                user's temp directory.
            kwargs: Additional keyword arguments to pass to the filelock.FileLock
        """
        path = Path(path)
        if tmpdir is None:
            tmp_dir = get_user_temp_dir()
        else:
            tmp_dir = Path(tmpdir)

        path_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()

        lock_path = tmp_dir / NEPS_LOCKFILE_DIR / f"{path_hash}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(lock_path, **kwargs)
        super().__init__(path=path, lock=lock)
