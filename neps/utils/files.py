"""Utilities for file operations."""

from __future__ import annotations

import dataclasses
import io
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import IO, Any, Literal

import yaml

try:
    from yaml import (
        CDumper as YamlDumper,  # type: ignore
        CSafeLoader as SafeLoader,  # type: ignore
    )
except ImportError:
    from yaml import SafeLoader, YamlDumper  # type: ignore


@contextmanager
def atomic_write(file_path: Path | str, *args: Any, **kwargs: Any) -> Iterator[IO]:
    """Write to a file atomically.

    This means that the file will be flushed to disk and explicitly ask the operating
    systems to sync the contents to disk. This ensures that other processes that read
    from this file should see the contents immediately.
    """
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


def serialize(
    data: Any,
    path: Path,
    *,
    check_serialized: bool = True,
    file_format: Literal["json", "yaml"] = "yaml",
    sort_keys: bool = True,
) -> None:
    """Serialize data to a yaml file."""
    if check_serialized:
        data = serializable_format(data)

    buf = io.StringIO()
    if file_format == "yaml":
        try:
            yaml.dump(data, buf, YamlDumper, sort_keys=sort_keys)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "Could not serialize to yaml! The object "
                f"{e.args[1]} of type {type(e.args[1])} is not."
            ) from e
    elif file_format == "json":
        import json

        json.dump(data, buf, sort_keys=sort_keys)
    else:
        raise ValueError(f"Unknown format: {file_format}")

    _str = buf.getvalue()
    path.write_text(_str)


def deserialize(
    path: Path | str,
    *,
    file_format: Literal["json", "yaml"] = "yaml",
) -> dict[str, Any]:
    """Deserialize data from a yaml file."""
    with Path(path).open("r") as file_stream:
        if file_format == "json":
            import json

            data = json.load(file_stream)
        elif file_format == "yaml":
            data = yaml.load(file_stream, SafeLoader)
        else:
            raise ValueError(f"Unknown format: {file_format}")

    if not isinstance(data, dict):
        raise TypeError(
            f"Deserialized data at {path} is not a dictionary!"
            f" Got {type(data)} instead.\n{data}"
        )

    return data
