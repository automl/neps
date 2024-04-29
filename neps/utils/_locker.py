from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, IO
from pathlib import Path

import portalocker as pl

EXCLUSIVE_NONE_BLOCKING = pl.LOCK_EX | pl.LOCK_NB


class Locker:
    FailedToAcquireLock = pl.exceptions.LockException

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_path.touch(exist_ok=True)

    @contextmanager
    def try_lock(self) -> Iterator[bool]:
        try:
            with self.acquire(fail_when_locked=True):
                yield True
        except self.FailedToAcquireLock:
            yield False

    def is_locked(self) -> bool:
        with self.try_lock() as acquired_lock:
            return not acquired_lock

    @contextmanager
    def __call__(
        self,
        poll: float = 1,
        *,
        timeout: float | None = None,
        fail_when_locked: bool = False,
    ) -> Iterator[IO]:
        with pl.Lock(
            self.lock_path,
            check_interval=poll,
            timeout=timeout,
            flags=EXCLUSIVE_NONE_BLOCKING,
            fail_when_locked=fail_when_locked,
        ) as fh:
            yield fh  # We almost never use it but nothing better to yield

    @contextmanager
    def acquire(
        self,
        poll: float = 1.0,
        *,
        timeout: float | None = None,
        fail_when_locked: bool = False,
    ) -> Iterator[IO]:
        with self(
            poll,
            timeout=timeout,
            fail_when_locked=fail_when_locked,
        ) as fh:
            yield fh
