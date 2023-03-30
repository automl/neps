from __future__ import annotations

import glob
import inspect
from functools import partial
from pathlib import Path
from typing import Any, Iterator

def non_empty_file(file_path: Path) -> bool:
    return file_path.exists() and file_path.stat().st_size != 0


def find_files(
    directory: Path,
    files: list[str],
    any_suffix: bool = False,
    check_nonempty: bool = False,
) -> Iterator[Path]:
    # OPTIM: Performing a recursive glob several times could be optimized
    # to only do this once. It can be pretty significant in large directories.
    for file_name in files:
        pattern = f"{directory.absolute()}/**/{file_name}"

        if any_suffix:
            pattern += "*"

        for f_path in glob.glob(pattern, recursive=True):
            path_found = Path(f_path)
            if path_found.is_file():
                if check_nonempty and not non_empty_file(path_found):
                    continue

                yield path_found


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
    kwargs: dict | None = None,
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
