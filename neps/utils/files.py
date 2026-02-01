"""Utilities for file operations."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import IO, Any, Literal, Protocol

import yaml

logger = logging.getLogger(__name__)

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


def load_and_merge_yamls(*paths: str | Path | IO[str]) -> dict[str, Any]:
    """Load and merge yaml files into a single dictionary.

    Raises:
        ValueError: If there are duplicate keys in the yaml files.
    """
    config: dict[str, Any] = {}
    for path in paths:
        match path:
            case str() | Path():
                with Path(path).open("r") as file:
                    read_config = yaml.safe_load(file)

            case _:
                read_config = yaml.safe_load(path)

        shared_keys = set(config) & set(read_config)

        if any(shared_keys):
            raise ValueError(f"Duplicate key(s) {shared_keys} in {paths}")

        config.update(read_config)

    return config


# ============================================================================
# Generic File Persistence System
# ============================================================================


class FileWriter(Protocol):
    """Protocol for writing content to disk."""

    def write(self, content: Any, file_path: Path | str) -> None:
        """Write content to disk.

        Args:
            content: Content to write.
            file_path: Path where content should be saved.
        """
        ...


class TextWriter:
    """Write text content to disk."""

    def write(self, content: str, file_path: Path | str) -> None:
        """Write text to file.

        Args:
            content: Text content to save.
            file_path: Path to save to (will use .txt extension).
        """
        file_path = Path(file_path).with_suffix(".txt")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            logger.debug(f"Wrote text to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write text to {file_path}: {e}")
            raise


class JsonWriter:
    """Write JSON-serializable content to disk."""

    def write(self, content: Any, file_path: Path | str) -> None:
        """Write JSON to file.

        Args:
            content: JSON-serializable object to save.
            file_path: Path to save to (will use .json extension).
        """
        file_path = Path(file_path).with_suffix(".json")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2, default=str)
            logger.debug(f"Wrote JSON to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write JSON to {file_path}: {e}")
            raise


class FigureWriter:
    """Write matplotlib Figure objects to disk."""

    def write(
        self,
        content: Any,
        file_path: Path | str,
        dpi: int = 100,
        fmt: str = "png",
    ) -> None:
        """Write matplotlib figure to file.

        Args:
            content: matplotlib Figure object.
            file_path: Path to save to (extension added based on fmt).
            dpi: Resolution for raster formats (default 100).
            fmt: Format to save as (default 'png'). Can also be 'pdf', 'svg', etc.
        """
        file_path = Path(file_path).with_suffix(f".{fmt}")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content.savefig(file_path, dpi=dpi, bbox_inches="tight")
            logger.debug(f"Wrote figure to {file_path}")
            # Clean up matplotlib resource
            try:
                import matplotlib.pyplot as plt

                plt.close(content)
            except ImportError:
                pass
        except Exception as e:
            logger.error(f"Failed to write figure to {file_path}: {e}")
            raise


class PickleWriter:
    """Write Python objects using pickle."""

    def write(self, content: Any, file_path: Path | str) -> None:
        """Write object using pickle.

        Args:
            content: Python object to pickle.
            file_path: Path to save to (will use .pkl extension).
        """
        import pickle

        file_path = Path(file_path).with_suffix(".pkl")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(content, f)
            logger.debug(f"Wrote pickle to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write pickle to {file_path}: {e}")
            raise


class BytesWriter:
    """Write raw bytes to disk."""

    def write(self, content: bytes, file_path: Path | str) -> None:
        """Write bytes to file.

        Args:
            content: Bytes content to save.
            file_path: Path to save to (will use .bin extension).
        """
        file_path = Path(file_path).with_suffix(".bin")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            logger.debug(f"Wrote bytes to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write bytes to {file_path}: {e}")
            raise


# Writer registry - maps content type to writer instance
FILE_WRITERS = {
    "text": TextWriter(),
    "json": JsonWriter(),
    "figure": FigureWriter(),
    "pickle": PickleWriter(),
    "bytes": BytesWriter(),
}


def get_file_writer(content_type: str) -> FileWriter:
    """Get the appropriate writer for a content type.

    Args:
        content_type: Type of content ('text', 'json', 'figure', 'pickle', 'bytes').

    Returns:
        Writer instance for the given type.

    Raises:
        ValueError: If content type is not supported.
    """
    if content_type not in FILE_WRITERS:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Supported types: {list(FILE_WRITERS.keys())}"
        )
    return FILE_WRITERS[content_type]
