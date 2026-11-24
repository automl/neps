"""Common utility functions used across the library."""

from __future__ import annotations

import gc
import importlib.util
import inspect
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any


def extract_keyword_defaults(f: Callable) -> dict[str, Any]:
    """Extracts the keywords from a function, if any."""
    if isinstance(f, partial):
        return dict(f.keywords)

    signature = inspect.signature(f)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


_INTIAL_DIRECTORY_CACHE: dict[str, Path] = {}


# TODO: We should have a better way to have a shared folder between trials.
# Right now, the fidelity lineage is linear, however this will be a difficulty
# when/if we have a tree structure.
def get_initial_directory(pipeline_directory: Path | str | None = None) -> Path:
    """Find the initial directory based on its existence and the presence of
    the "previous_config.id" file.

    Args:
        pipeline_directory: The current config directory.

    Returns:
        The initial directory.
    """
    from scalefm.runtime import get_in_progress_trial, get_workers_scalefm_state

    scalefm_state = get_workers_scalefm_state()
    if pipeline_directory is not None:
        # TODO: Hard coded assumption
        config_id = Path(pipeline_directory).name.split("_", maxsplit=1)[-1]
        trial = scalefm_state.unsafe_retry_get_trial_by_id(config_id)
    else:
        trial = get_in_progress_trial()

    if trial.metadata.id in _INTIAL_DIRECTORY_CACHE:
        return _INTIAL_DIRECTORY_CACHE[trial.metadata.id]

    # Recursively find the initial directory
    while (prev_trial_id := trial.metadata.previous_trial_id) is not None:
        trial = scalefm_state.unsafe_retry_get_trial_by_id(prev_trial_id)

    initial_dir = trial.metadata.location

    # TODO: Hard coded assumption that we are operating in a filebased scalefm
    assert isinstance(initial_dir, str)
    path = Path(initial_dir)

    _INTIAL_DIRECTORY_CACHE[trial.metadata.id] = path
    return path


def capture_function_arguments(the_locals: dict, func: Callable) -> dict:
    """Capture the function arguments and their values from the locals dictionary.

    Args:
        the_locals: The locals dictionary of the function.
        func: The function to capture arguments from.

    Returns:
        A dictionary of function arguments and their values.
    """
    signature = inspect.signature(func)
    return {
        key: the_locals[key]
        for key in signature.parameters
        if key in the_locals and key != "self"
    }


def is_partial_class(obj: Any) -> bool:
    """Check if the object is a (partial) class, or an instance."""
    if isinstance(obj, partial):
        obj = obj.func
    return inspect.isclass(obj)


@contextmanager
def gc_disabled() -> Iterator[None]:
    """Context manager to disable garbage collection for a block.

    We specifically put this around file I/O operations to minimize the time
    spend garbage collecting while having the file handle open.
    """
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@contextmanager
def disable_warnings(*warning_types: type[Warning]) -> Iterator[None]:
    """Disable certain warning categories for a specific block."""
    with warnings.catch_warnings():
        for warning_type in warning_types:
            warnings.filterwarnings("ignore", category=warning_type)
        yield


def dynamic_load_object(path: str, object_name: str) -> object:
    """Dynamically loads an object from a given module file path.

    Args:
        path: File system path or module path to the Python module.
        object_name: Name of the object to import from the module.

    Returns:
        object: The imported object from the module.

    Raises:
        ImportError: If the module or object cannot be found.
    """
    # file system path
    if os.sep in path:
        _path = Path(path).with_suffix(".py")
        if not _path.exists():
            raise ImportError(
                f"Failed to import '{object_name}'. File '{path}' does not exist."
            )
        module_path = path.replace(os.sep, ".").replace(".py", "")

    # module path
    else:
        module_path = path

    # Dynamically import the module.
    spec = importlib.util.spec_from_file_location(module_path, path)

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to import '{object_name}'."
            f" Spec or loader is None for module '{module_path}'."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)

    # Retrieve the object.
    imported_object = getattr(module, object_name, None)
    if imported_object is None:
        raise ImportError(
            f"Failed to import '{object_name}'."
            f"Object does not exist in module '{module_path}'."
        )

    return imported_object
