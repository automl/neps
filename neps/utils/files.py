"""Utilities for file operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def _serializable_format(data: Any) -> Any:
    if hasattr(data, "serialize"):
        return _serializable_format(data.serialize())

    if isinstance(data, Mapping):
        return {key: _serializable_format(val) for key, val in data.items()}

    if not isinstance(data, str) and isinstance(data, Iterable):
        return [_serializable_format(val) for val in data]

    if type(data).__module__ in ["numpy", "torch"]:
        data = data.tolist()  # type: ignore
        if type(data).__module__ == "numpy":
            data = data.item()

        return _serializable_format(data)

    return data


def serialize(data: Any, path: Path | str, *, sort_keys: bool = True) -> None:
    """Serialize data to a yaml file."""
    data = _serializable_format(data)
    path = Path(path)
    with path.open("w") as file_stream:
        try:
            return yaml.safe_dump(data, file_stream, sort_keys=sort_keys)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "Could not serialize to yaml! The object "
                f"{e.args[1]} of type {type(e.args[1])} is not."
            ) from e


def deserialize(path: Path | str) -> dict[str, Any]:
    """Deserialize data from a yaml file."""
    with Path(path).open("r") as file_stream:
        return yaml.full_load(file_stream)


def empty_file(file_path: Path) -> bool:
    """Check if a file does not exist, or if it does, if it is empty."""
    return not file_path.exists() or file_path.stat().st_size <= 0
