from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import yaml

from typing import Iterable, Mapping


def get_data_representation(data: Any):
    """Common data representations. Other specific types should be handled
    by the user in his Parameter class."""
    if hasattr(data, "serialize"):
        return get_data_representation(data.serialize())

    if isinstance(data, Mapping):
        return {key: get_data_representation(val) for key, val in data.items()}

    if not isinstance(data, str) and isinstance(data, Iterable):
        return [get_data_representation(val) for val in data]

    if type(data).__module__ in ["numpy", "torch"]:
        data = data.tolist()  # type: ignore
        if type(data).__module__ == "numpy":
            data = data.item()
        return get_data_representation(data)

    return data


def serialize(data: Any, path: Path | str, sort_keys: bool = True) -> None:
    data = get_data_representation(data)
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
    with Path(path).open("r") as file_stream:
        return yaml.full_load(file_stream)


def empty_file(file_path: Path) -> bool:
    return not file_path.exists() or file_path.stat().st_size <= 0


def find_files(
    directory: Path,
    files: Iterable[str],
    any_suffix: bool = False,
    check_nonempty: bool = False,
) -> list[Path]:
    found_paths = []
    for file_name in files:
        pattern = f"{directory.absolute()}/**/{file_name}"
        if any_suffix:
            pattern += "*"
        for f_path in glob.glob(pattern, recursive=True):
            path_found = Path(f_path)
            if path_found.is_file():
                if check_nonempty and empty_file(path_found):
                    continue
                found_paths.append(path_found)
    return found_paths
