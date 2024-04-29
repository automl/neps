"""Primitive types to be used in NePS or consumers of NePS."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Union
from typing_extensions import TypeAlias

# NOTE: `SearchSpace` is also one
ConfigLike: TypeAlias = Mapping[str, Any]

# TODO(eddiebergman): We can turn this to an enum at some
# point to prevent having to isinstance and str match
ERROR: TypeAlias = Literal["error"]

POST_EVAL_HOOK_SIGNATURE: TypeAlias = Callable[
    [
        ConfigLike,
        str,
        Path,
        Union[Dict[str, Any], ERROR],
        logging.Logger,
    ],
    None,
]


# TODO(eddiebergman): Ideally, use `Trial` objects which can carry a lot more
# useful information to optimizers than the below dataclass. Would be a follow up
# refactor.
@dataclass
class ConfigResult:
    """Primary class through which optimizers recieve results."""

    id: str
    """Unique identifier for the configuration."""

    # TODO(eddiebergman): Check if this is a `SearchSpace` everywhere and change to
    # that if so
    config: ConfigLike
    """Configuration that was evaluated."""

    # TODO(eddiebergman): Check about using a `TypedDict` here since I'm pretty sure
    # there's always a "loss" key
    result: dict
    """Some dictionary of results."""

    metadata: dict
    """Any additional data to store with this config and result."""


# NOTE: Please try to avoid using this class and prefer a dict if its dynamic
# or make a dataclass if the fields are known and are static
class AttrDict(dict):
    """Dictionary that allows access to keys as attributes."""

    def __init__(self, *args, **kwargs):
        """Initialize like a dict."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self
