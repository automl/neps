from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from typing_extensions import Self, override

from neps.state.resource import Resource
from neps.utils.files import deserialize, serialize


@dataclass
class OptimizerState(Resource):
    directory: Path
    # TODO(eddiebergman): What are the common keywords
    # we can use that don't have to be crammed into mapping
    state: Mapping[str, Any]
    info: Mapping[str, Any]
    info_path: Path = field(init=False)
    state_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.info_path = self.directory / ".optimizer_info.yaml"
        self.state_path = self.directory / ".optimizer_state.yaml"

    @override
    @classmethod
    def load(cls, directory: Path) -> Self:
        info_path = directory / ".optimizer_info.yaml"
        state_path = directory / ".optimizer_state.yaml"
        info = deserialize(info_path)
        state = deserialize(state_path)
        return cls(directory=directory, info=info, state=state)

    @override
    def commit(self) -> None:
        serialize(self.info, self.info_path)
        serialize(self.state, self.state_path)

    @override
    def update(self) -> None:
        self.info = deserialize(self.info_path)
