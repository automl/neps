"""Utilities for file operations."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import IO, Any

import yaml

try:
    from yaml import (
        CSafeDumper as SafeDumper,  # type: ignore
        CSafeLoader as SafeLoader,  # type: ignore
    )
except ImportError:
    from yaml import SafeDumper, SafeLoader  # type: ignore


@contextmanager
def atomic_write(file_path: Path | str, *args: Any, **kwargs: Any) -> Iterator[IO]:
    with open(file_path, *args, **kwargs) as file_stream:  # noqa: PTH123
        yield file_stream
        file_stream.flush()
        os.fsync(file_stream.fileno())
        file_stream.close()


def serializable_format(data: Any) -> Any:  # noqa: PLR0911
    """Format data to be serializable."""
    if hasattr(data, "serialize"):
        return serializable_format(data.serialize())

    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return serializable_format(dataclasses.asdict(data))  # type: ignore

    if isinstance(data, Exception):
        return str(data)

    if isinstance(data, Enum):
        return data.value

    if isinstance(data, Mapping):
        return {key: serializable_format(val) for key, val in data.items()}

    if not isinstance(data, str) and isinstance(data, Iterable):
        return [serializable_format(val) for val in data]

    if type(data).__module__ in ["numpy", "torch"]:
        data = data.tolist()  # type: ignore
        if type(data).__module__ == "numpy":
            data = data.item()

        return serializable_format(data)

    return data


def serialize(data: Any, path: Path | str, *, sort_keys: bool = True) -> None:
    """Serialize data to a yaml file."""
    data = serializable_format(data)
    path = Path(path)
    with atomic_write(path, "w") as file_stream:
        try:
            return yaml.dump(data, file_stream, SafeDumper, sort_keys=sort_keys)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "Could not serialize to yaml! The object "
                f"{e.args[1]} of type {type(e.args[1])} is not."
            ) from e


def deserialize(path: Path | str) -> dict[str, Any]:
    """Deserialize data from a yaml file."""
    with Path(path).open("r") as file_stream:
        data = yaml.load(file_stream, SafeLoader)

    if not isinstance(data, dict):
        raise TypeError(
            f"Deserialized data at {path} is not a dictionary!"
            f" Got {type(data)} instead.\n{data}"
        )

    return data
