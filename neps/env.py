"""Environment variable parsing for the state."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal, TypeVar

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


def yaml_or_json(e: str) -> Literal["yaml", "json"]:
    """Check if an environment variable is either yaml or json."""
    if e.lower() in ("yaml", "json"):
        return e.lower()  # type: ignore
    raise ValueError(f"Expected 'yaml' or 'json', got '{e}'.")


LINUX_FILELOCK_FUNCTION = get_env(
    "NEPS_LINUX_FILELOCK_FUNCTION",
    parse=str,
    default="lockf",
)
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
FS_SYNC_GRACE_BASE = get_env(
    "NEPS_FS_SYNC_GRACE_BASE",
    parse=float,
    default=0.00,  # Keep it low initially to not punish synced os
)
FS_SYNC_GRACE_INC = get_env(
    "NEPS_FS_SYNC_GRACE_INC",
    parse=float,
    default=0.1,
)

# NOTE: We want this to be greater than the trials filelock, so that
# anything requesting to just update the trials is more likely to obtain it
# as those operations tend to be faster than something that requires optimizer
# state.
STATE_FILELOCK_POLL = get_env(
    "NEPS_STATE_FILELOCK_POLL",
    parse=float,
    default=0.20,
)
STATE_FILELOCK_TIMEOUT = get_env(
    "NEPS_STATE_FILELOCK_TIMEOUT",
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
TRIAL_CACHE_MAX_UPDATES_BEFORE_CONSOLIDATION = get_env(
    "NEPS_TRIAL_CACHE_MAX_UPDATES_BEFORE_CONSOLIDATION",
    parse=int,
    default=30,
)
CONFIG_SERIALIZE_FORMAT: Literal["yaml", "json"] = get_env(  # type: ignore
    "NEPS_CONFIG_SERIALIZE_FORMAT",
    parse=yaml_or_json,
    default="yaml",
)
