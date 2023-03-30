from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Mapping, TypeVar
from contextlib import contextmanager

from more_itertools import first
from typing_extensions import Literal, Self

from ._locker import Locker
from .serialization import Serializer
from .utils import find_files, non_empty_file

DeserializedConfig = TypeVar("DeserializedConfig")


class Config:
    def __init__(self, path: Path, serializer: Serializer | None = None) -> None:
        if serializer is None:
            serializer = Serializer.default()

        self.path = path
        self.serializer = serializer

    @dataclass
    class Result(Generic[DeserializedConfig]):
        config: DeserializedConfig
        result: Literal["error"] | str | float | int | Mapping[str, Any]
        metadata: Mapping[str, Any]
        disk: Config

    def as_result(
        self,
        config_deserializer: Callable[[Mapping[str, Any]], DeserializedConfig],
    ) -> Config.Result[DeserializedConfig]:
        config = self.raw_config()

        if config is None:
            raise ValueError(f"No config for this configuration {self=}")

        result = self.result()
        metadata = self.metadata()

        if result is None or metadata is None:
            raise ValueError(
                f"No result/metadata for this configuration {config=} \n {self=}"
            )

        parsed_config = config_deserializer(config)
        return Config.Result(parsed_config, result, metadata, disk=self)

    @cached_property
    def id(self) -> str:
        return self.path.name[len("config_") :]

    @cached_property
    def previous_pipeline_id(self) -> str | None:
        if not self.previous_config_id_file.exists():
            return None

        return self.previous_config_id_file.read_text()

    @cached_property
    def previous_pipeline_dir(self) -> Path | None:
        if not self.previous_config_dir_file.exists():
            return None

        return Path(self.previous_config_dir_file.read_text())

    @cached_property
    def config_file(self) -> Path:
        suffix = self.serializer.SUFFIX
        return (self.path / "config").with_suffix(suffix)

    @cached_property
    def previous_config_id_file(self) -> Path:
        return self.path / "previous_config.id"

    @cached_property
    def previous_config_dir_file(self) -> Path:
        return self.path / "previous_config.dir"

    @cached_property
    def result_file(self) -> Path:
        suffix = self.serializer.SUFFIX
        return (self.path / "result").with_suffix(suffix)

    @cached_property
    def metadata_file(self) -> Path:
        suffix = self.serializer.SUFFIX
        return (self.path / "metadata").with_suffix(suffix)

    @cached_property
    def lockfile(self) -> Path:
        return self.path / ".config_lock"

    def fresh(self) -> Self:
        return self.__class__(self.path, serializer=self.serializer)

    def has_result(self) -> bool:
        return non_empty_file(self.result_file)

    def has_config(self) -> bool:
        return non_empty_file(self.config_file)

    def has_metadata(self) -> bool:
        return non_empty_file(self.config_file)

    def remove(self) -> None:
        shutil.rmtree(str(self.path))

    def result(
        self,
    ) -> Literal["error"] | str | float | int | Mapping[str, Any] | None:
        if not self.has_result():
            return None

        return self.serializer.load(self.result_file)

    def raw_config(self) -> Mapping[str, Any] | None:
        if not self.has_config():
            return None

        return self.serializer.load(self.config_file)

    def metadata(self) -> dict[str, Any] | None:
        if not self.has_metadata():
            return None

        return self.serializer.load(self.metadata_file)

    def broken(self) -> bool:
        return not self.has_config() or not self.has_result()

    def existing_config_file(self) -> Path | None:
        itr = find_files(self.path, ["config"], any_suffix=True, check_nonempty=True)
        return first(itr, None)

    @property
    def locked(self) -> bool:
        return Locker.local(self.config_file, suffix=".config_lock").locked

    @contextmanager
    def lock(
        self, active: bool = True, *, timeout: float | None = None
    ) -> Iterator[bool]:
        """Return True if the state is active, False otherwise"""
        if active:
            lock = Locker.local(self.lockfile)
            with lock.lock(timeout=timeout) as acquired:
                yield acquired
        else:
            yield True

    def status(self) -> Config.Status:
        if self.has_result():
            return Config.Status.COMPLETE

        if self.has_config():
            if self.locked:
                return Config.Status.ACTIVE
            else:
                return Config.Status.FREE

        if self.broken():
            return Config.Status.BROKEN

        return Config.Status.UNKNOWN

    class Status(Enum):
        COMPLETE = auto()
        """The configuration has been evaluated and the result is available."""

        FREE = auto()
        """The configuration is there with no lock."""

        ACTIVE = auto()
        """The configuration is there and locked."""

        BROKEN = auto()
        """The directory exists but somehow the config is not correct."""

        UNKNOWN = auto()
        """Something is wrong."""
