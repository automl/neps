from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any, Literal, Callable, Union, Dict
from typing_extensions import TypeAlias
from dataclasses import dataclass
import logging

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
    id: str
    config: ConfigLike
    result: dict
    metadata: dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
