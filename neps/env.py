"""Environment variable parsing for the state."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
V = TypeVar("V")

ENV_VARS_USED: dict[str, tuple[Any, Any]] = {}


def get_env(key: str, parse: Callable[[str], T], default: V) -> T | V:
    """Get an environment variable or return a default value."""
    if (e := os.environ.get(key)) is not None:
        value = parse(e)
        ENV_VARS_USED[key] = (e, value)
        return value

    ENV_VARS_USED[key] = (default, default)
    return default


def is_nullable(e: str) -> bool:
    """Check if an environment variable is nullable."""
    return e.lower() in ("none", "n", "null")


LINUX_FILELOCK_FUNCTION = get_env(
    "NEPS_LINUX_FILELOCK_FUNCTION",
    parse=str,
    default="lockf",
)
assert LINUX_FILELOCK_FUNCTION in ("lockf", "flock")

MAX_RETRIES_GET_NEXT_TRIAL = get_env(
    "NEPS_MAX_RETRIES_GET_NEXT_TRIAL",
    parse=int,
    default=10,
)
MAX_RETRIES_SET_EVALUATING = get_env(
    "NEPS_MAX_RETRIES_SET_EVALUATING",
    parse=int,
    default=10,
)
MAX_RETRIES_CREATE_LOAD_STATE = get_env(
    "NEPS_MAX_RETRIES_CREATE_LOAD_STATE",
    parse=int,
    default=10,
)
MAX_RETRIES_WORKER_CHECK_SHOULD_STOP = get_env(
    "NEPS_MAX_RETRIES_WORKER_CHECK_SHOULD_STOP",
    parse=int,
    default=3,
)
TRIAL_FILELOCK_POLL = get_env(
    "NEPS_TRIAL_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
TRIAL_FILELOCK_TIMEOUT = get_env(
    "NEPS_TRIAL_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=120,
)

SEED_SNAPSHOT_FILELOCK_POLL = get_env(
    "NEPS_SEED_SNAPSHOT_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
SEED_SNAPSHOT_FILELOCK_TIMEOUT = get_env(
    "NEPS_SEED_SNAPSHOT_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=120,
)

OPTIMIZER_INFO_FILELOCK_POLL = get_env(
    "NEPS_OPTIMIZER_INFO_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
OPTIMIZER_INFO_FILELOCK_TIMEOUT = get_env(
    "NEPS_OPTIMIZER_INFO_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=120,
)

OPTIMIZER_STATE_FILELOCK_POLL = get_env(
    "NEPS_OPTIMIZER_STATE_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
OPTIMIZER_STATE_FILELOCK_TIMEOUT = get_env(
    "NEPS_OPTIMIZER_STATE_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=120,
)

GLOBAL_ERR_FILELOCK_POLL = get_env(
    "NEPS_GLOBAL_ERR_FILELOCK_POLL",
    parse=float,
    default=0.05,
)
GLOBAL_ERR_FILELOCK_TIMEOUT = get_env(
    "NEPS_GLOBAL_ERR_FILELOCK_TIMEOUT",
    parse=lambda e: None if is_nullable(e) else float(e),
    default=120,
)
