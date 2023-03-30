from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, TypeVar

import yaml
from typing_extensions import Protocol, TypeAlias, runtime_checkable

R = TypeVar("R")


def identity(x: dict[str, Any]) -> dict[str, Any]:
    return x


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> SerializableType:
        ...


@runtime_checkable
class Listable(Protocol):
    def tolist(self) -> list[SerializableType]:
        ...


SerializableType: TypeAlias = (
    Mapping[str, "SerializableType"]
    | Iterable["SerializableType"]
    | Listable
    | Serializable
    | str
    | int
    | float
)


def get_data_representation(data: SerializableType):
    """Common data representations. Other specific types should be handled
    by the user in his Parameter class."""
    if isinstance(data, dict):
        return {key: get_data_representation(val) for key, val in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [get_data_representation(val) for val in data]
    elif isinstance(data, Listable):
        data = data.tolist()
        if type(data).__module__ == "numpy":
            data = data.item()  # type: ignore
        return get_data_representation(data)
    elif isinstance(data, Serializable):
        return get_data_representation(data.serialize())
    else:
        return data


class Serializer(ABC):
    SUFFIX: ClassVar[str]

    def load(self, path: Path | str, add_suffix: bool = True) -> dict[str, Any]:
        path = Path(path)
        if add_suffix:
            path = path.with_suffix(self.SUFFIX)

        return self._load(path)

    def dump(
        self,
        data: SerializableType,
        path: Path | str,
        add_suffix: bool = True,
        pre_serialize: bool = True,
    ) -> None:
        if pre_serialize:
            data = get_data_representation(data)

        path = Path(path)
        if add_suffix:
            path = path.with_suffix(self.SUFFIX)

        self._dump(data, path)

    @abstractmethod
    def _load(self, path: Path) -> dict[str, Any]:
        ...

    @abstractmethod
    def _dump(self, data: Any, path: Path) -> None:
        ...

    @classmethod
    def default(cls) -> YamlSerializer:
        return YamlSerializer()


class YamlSerializer(Serializer):
    SUFFIX = ".yaml"

    def _load(self, path: Path) -> dict[str, Any]:
        with path.open("r") as fstream:
            return yaml.full_load(fstream)

    def _dump(self, data: SerializableType, path: Path) -> None:
        try:
            with open(path, "w") as fstream:
                yaml.safe_dump(data, fstream)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "You should return objects that are JSON-serializable. The object "
                f"{e.args[1]} of type {type(e.args[1])} is not."
            ) from e
