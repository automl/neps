from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator
from typing_extensions import Protocol

import portalocker as pl

FILELOCK_EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB


class Locker(Protocol):
    """A versioned serializer that can read and write a resource to disk."""

    @contextmanager
    def lock(self) -> Iterator[None]: ...

    def is_locked(self) -> bool: ...


@dataclass
class FileLocker(Locker):
    FailedToAcquireLock = pl.exceptions.LockException

    lock_path: Path
    poll: float = 0.1
    timeout: float | None = None

    def is_locked(self) -> bool:
        if not self.lock_path.exists():
            return False
        try:
            with self.lock(fail_if_locked=True):
                pass
            return False
        except self.FailedToAcquireLock:
            return True

    @contextmanager
    def lock(
        self,
        *,
        fail_if_locked: bool = False,
    ) -> Iterator[IO]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        with pl.Lock(
            self.lock_path,
            check_interval=self.poll,
            timeout=self.timeout,
            flags=FILELOCK_EXCLUSIVE_NONE_BLOCKING,
            fail_when_locked=fail_if_locked,
        ) as fh:
            yield fh  # We almost never use it but nothing better to yield
