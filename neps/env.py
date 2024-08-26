"""Environment variable parsing for the state."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
V = TypeVar("V")


def get_env(key: str, parse: Callable[[str], T], default: V) -> T | V:
    """Get an environment variable or return a default value."""
    if (e := os.environ.get(key)) is not None:
        return parse(e)

    return default


def is_nullable(e: str) -> bool:
    """Check if an environment variable is nullable."""
    return e.lower() in ("none", "n", "null")


TRIAL_FILELOCK_POLL = get_env(
    "NEPS_TRIAL_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
TRIAL_FILELOCK_TIMEOUT = get_env(
    "NEPS_TRIAL_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)

JOBQUEUE_FILELOCK_POLL = get_env(
    "NEPS_JOBQUEUE_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
JOBQUEUE_FILELOCK_TIMEOUT = get_env(
    "NEPS_JOBQUEUE_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)

SEED_SNAPSHOT_FILELOCK_POLL = get_env(
    "NEPS_SEED_SNAPSHOT_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
SEED_SNAPSHOT_FILELOCK_TIMEOUT = get_env(
    "NEPS_SEED_SNAPSHOT_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)

OPTIMIZER_INFO_FILELOCK_POLL = get_env(
    "NEPS_OPTIMIZER_INFO_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
OPTIMIZER_INFO_FILELOCK_TIMEOUT = get_env(
    "NEPS_OPTIMIZER_INFO_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)

OPTIMIZER_STATE_FILELOCK_POLL = get_env(
    "NEPS_OPTIMIZER_STATE_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
OPTIMIZER_STATE_FILELOCK_TIMEOUT = get_env(
    "NEPS_OPTIMIZER_STATE_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)

GLOBAL_ERR_FILELOCK_POLL = get_env(
    "NEPS_GLOBAL_ERR_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
GLOBAL_ERR_FILELOCK_TIMEOUT = get_env(
    "NEPS_GLOBAL_ERR_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=None,
)
