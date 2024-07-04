from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar
from typing_extensions import Protocol, override

from neps.utils.files import empty_file

T = TypeVar("T")
K = TypeVar("K")


class VersionedStore(Protocol[T, K]):
    """A versioned serializer that can read and write a resource to disk."""

    def current_version(self) -> str | None: ...
    def key(self) -> K: ...
    def get(self) -> tuple[T, str]: ...
    def put(self, data: T, version: str | None, /) -> str: ...
    def exists(self) -> bool: ...


@dataclass(kw_only=True)
class VersionedDirectoryStore(VersionedStore[T, Path]):
    directory: Path
    read: Callable[[Path], T]
    write: Callable[[T, Path], None]
    version_filename: str = ".version"

    @property
    def version_file(self) -> Path:
        return self.directory / self.version_filename

    @override
    def key(self) -> Path:
        return self.directory

    @override
    def current_version(self) -> str | None:
        if empty_file(self.version_file):
            return None

        return self.version_file.read_text()

    @override
    def exists(self) -> bool:
        return self.current_version() is not None

    @override
    def get(self) -> tuple[T, str]:
        current_version = self.current_version()
        if current_version is None:
            raise FileNotFoundError(f"Resource does not exist at '{self.key()}'")

        data = self.read(self.directory)
        return data, current_version

    @override
    def put(self, data: T, previous_version: str | None) -> str:
        current_version = self.current_version()
        if previous_version != current_version:
            raise RuntimeError(
                f"Version mismatch: {previous_version} != {current_version}"
            )

        self.write(data, self.directory)

        new_version = uuid.uuid4().hex
        self.version_file.write_text(new_version)
        return new_version
