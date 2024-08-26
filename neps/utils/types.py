"""Primitive types to be used in NePS or consumers of NePS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.trial import Trial

# TODO(eddiebergman): We can turn this to an enum at some
# point to prevent having to isinstance and str match
ERROR: TypeAlias = Literal["error"]
Number: TypeAlias = int | float | np.number
ConfigID: TypeAlias = str
RawConfig: TypeAlias = Mapping[str, Any]
Metadata: TypeAlias = dict[str, Any]
ResultDict: TypeAlias = Mapping[str, Any]

# NOTE(eddiebergman): Getting types for scipy distributions sucks
# this is more backwards compatible and easier to work with
TruncNorm: TypeAlias = Any


class _NotSet:
    def __repr__(self) -> str:
        return "NotSet"


NotSet = _NotSet()

f64 = np.float64
i64 = np.int64


# TODO(eddiebergman): Ideally, use `Trial` objects which can carry a lot more
# useful information to optimizers than the below dataclass. Would be a follow up
# refactor.
@dataclass
class ConfigResult:
    """Primary class through which optimizers recieve results."""

    id: str
    """Unique identifier for the configuration."""

    config: SearchSpace
    """Configuration that was evaluated."""

    result: Trial.Report | ResultDict | ERROR
    """Some dictionary of results."""

    metadata: dict
    """Any additional data to store with this config and result."""


# TODO(eddiebergman): This is a hack because status.py expects a `ConfigResult`
# where the `config` is a dict config (`RawConfig`), while all the optimizers
# expect a `ConfigResult` where the `config` is a `SearchSpace`. Ideally we
# just rework status to use `Trial` and `Report` directly as they contain a lot more
# information.
@dataclass
class _ConfigResultForStats:
    id: str
    config: RawConfig
    result: ResultDict | ERROR
    metadata: dict

    @property
    def loss(self) -> float | ERROR:
        if isinstance(self.result, dict):
            return float(self.result["loss"])
        return "error"


# NOTE: Please try to avoid using this class and prefer a dict if its dynamic
# or make a dataclass if the fields are known and are static
class AttrDict(dict):
    """Dictionary that allows access to keys as attributes."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize like a dict."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self
