from __future__ import annotations

import glob
import inspect
import json
import logging
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable

import dill
import yaml


def non_empty_file(file_path: Path) -> bool:
    return file_path.exists() and file_path.stat().st_size != 0


def find_files(
    directory: Path, files: list[str], any_suffix=False, check_nonempty=False
) -> list[Path]:
    found_paths = []
    for file_name in files:
        pattern = f"{directory.absolute()}/**/{file_name}"
        if any_suffix:
            pattern += "*"
        for f_path in glob.glob(pattern, recursive=True):
            path_found = Path(f_path)
            if path_found.is_file():
                if check_nonempty and not non_empty_file(path_found):
                    continue
                found_paths.append(path_found)
    return found_paths


# Serializers


def init_serializer(serializer=None, read_files_in=None, logger=None):
    if logger is None:
        logger = logging.getLogger("metahyper")
    if serializer is None:
        if read_files_in is not None:
            for name, serializer_cls in SerializerMapping.items():
                data_files = [
                    f".optimizer_state{serializer_cls.SUFFIX}",
                    f"config{serializer_cls.SUFFIX}",
                    f"result{serializer_cls.SUFFIX}",
                ]
                if find_files(read_files_in, data_files):
                    serializer = name
                    logger.debug(f"Auto-detected {name} format for serializer")
                    break
        if serializer is None:
            serializer = "json"
            logger.debug(f"Will use the {serializer} serializer as a default")

    return instance_from_map(SerializerMapping, serializer, "serializer")


def get_data_representation(data: Any):
    """Common data representations. Other specific types should be handled
    by the user in his Parameter class."""
    if isinstance(data, dict):
        return {key: get_data_representation(val) for key, val in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [get_data_representation(val) for val in data]
    elif type(data).__module__ in ["numpy", "torch"]:
        data = data.tolist()
        if type(data).__module__ == "numpy":
            data = data.item()
        return get_data_representation(data)
    elif hasattr(data, "serialize"):
        return get_data_representation(data.serialize())
    else:
        return data


class DataSerializer:
    SUFFIX = ""
    PRE_SERIALIZE = True

    def __init__(self, config_loader: Callable | None = None):
        self.config_loader = config_loader or (lambda x: x)

    @abstractmethod
    def _load_from(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def _dump_to(self, data: Any, path: str):
        raise NotImplementedError

    def load(self, path: Path | str, add_suffix=True):
        path = str(path)
        if add_suffix and Path(path).suffix != self.SUFFIX:
            path = path + self.SUFFIX
        return self._load_from(path)

    def dump(self, data: Any, path: Path | str, add_suffix=True):
        if self.PRE_SERIALIZE:
            data = get_data_representation(data)
        path = str(path)
        if add_suffix and Path(path).suffix != self.SUFFIX:
            path = path + self.SUFFIX
        self._dump_to(data, path)

    def load_config(self, path: Path | str):
        if self.PRE_SERIALIZE:
            return self.config_loader(self.load(path))
        return self.load(path)


class DillSerializer(DataSerializer):
    SUFFIX = ".dill"
    PRE_SERIALIZE = False

    def _load_from(self, path: str):
        with open(path, "rb") as file_stream:
            return dill.load(file_stream)

    def _dump_to(self, data: Any, path: str):
        with open(path, "wb") as file_stream:
            return dill.dump(data, file_stream)


class JsonSerializer(DataSerializer):
    SUFFIX = ".json"

    def _load_from(self, path: str):
        with open(path) as file_stream:
            return json.load(file_stream)

    def _dump_to(self, data: Any, path: str):
        with open(path, "w") as file_stream:
            return json.dump(data, file_stream)


class YamlSerializer(DataSerializer):
    SUFFIX = ".yaml"

    def _load_from(self, path: str):
        with open(path) as file_stream:
            return yaml.full_load(file_stream)

    def _dump_to(self, data: Any, path: str):
        with open(path, "w") as file_stream:
            try:
                return yaml.safe_dump(data, file_stream)
            except yaml.representer.RepresenterError as e:
                raise TypeError(
                    f"You should return objects that are JSON-serializable. The object {e.args[1]} of type {type(e.args[1])} is not."
                ) from e


SerializerMapping = {
    "yaml": YamlSerializer,
    "json": JsonSerializer,
    "dill": DillSerializer,
}

# Mappings


def is_partial_class(obj):
    """Check if the object is a (partial) class, or an instance"""
    if isinstance(obj, partial):
        obj = obj.func
    return inspect.isclass(obj)


def instance_from_map(
    mapping: dict[str, Any],
    request: str | list | tuple | Any,
    name: str = "mapping",
    allow_any: bool = True,
    as_class: bool = False,
    kwargs: dict = None,
):
    """Get an instance of an class from a mapping.

    Arguments:
        mapping: Mapping from string keys to classes or instances
        request: A key from the mapping. If allow_any is True, could also be an
            object or a class, to use a custom object.
        name: Name of the mapping used in error messages
        allow_any: If set to True, allows using custom classes/objects.
        as_class: If the class should be returned without beeing instanciated
        kwargs: Arguments used for the new instance, if created. Its purpose is
            to serve at default arguments if the user doesn't built the object.

    Raises:
        ValueError: if the request is invalid (not a string if allow_any is False),
            or invalid key.
    """

    # Split arguments of the form (request, kwargs)
    args_dict = kwargs or {}
    if isinstance(request, tuple) or isinstance(request, list):
        if len(request) != 2:
            raise ValueError(
                "When building an instance and specifying arguments, "
                "you should give a pair (class, arguments)"
            )
        request, req_args_dict = request
        if not isinstance(req_args_dict, dict):
            raise ValueError("The arguments should be given as a dictionary")
        args_dict = {**args_dict, **req_args_dict}

    # Then, get the class/instance from the request
    if isinstance(request, str):
        if request in mapping:
            instance = mapping[request]
        else:
            raise ValueError(f"{request} doesn't exists for {name}")
    elif allow_any:
        instance = request
    else:
        raise ValueError(f"Object {request} invalid key for {name}")

    # Check if the request is a class if it is mandatory
    if (args_dict or as_class) and not is_partial_class(instance):
        raise ValueError(
            f"{instance} is not a class and can't be used with additional arguments"
        )

    # Give the arguments to the class
    if args_dict:
        instance = partial(instance, **args_dict)

    # Return the class / instance
    if as_class:
        return instance
    if is_partial_class(instance):
        try:
            instance = instance()
        except TypeError as e:
            raise TypeError(f"{e} when calling {instance} with {args_dict}") from e
    return instance
