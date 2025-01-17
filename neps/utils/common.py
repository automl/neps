"""Common utility functions used across the library."""

from __future__ import annotations

import gc
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import torch


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


# TODO(eddiebergman): I feel like this function should throw an error if it can't
# find anything to load, rather than returning None. In this case, we should provide
# the user an easy way to check if there is some previous checkpoint to load.
def load_checkpoint(
    directory: Path | str | None = None,
    checkpoint_name: str = "checkpoint",
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict | None:
    """Load a checkpoint and return the model state_dict and checkpoint values.

    Args:
        directory: Directory where the checkpoint is located.
        checkpoint_name: The name of the checkpoint file.
        model: The PyTorch model to load.
        optimizer: The optimizer to load.

    Returns:
        A dictionary containing the checkpoint values, or None if the checkpoint file
        does not exist hence no checkpointing was previously done.
    """
    from neps.runtime import get_in_progress_trial

    if directory is None:
        trial = get_in_progress_trial()
        directory = trial.metadata.previous_trial_location
        if directory is None:
            return None
        assert isinstance(directory, str)

    directory = Path(directory)
    checkpoint_path = (directory / checkpoint_name).with_suffix(".pth")

    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, weights_only=True)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint  # type: ignore


def save_checkpoint(
    directory: Path | str | None = None,
    checkpoint_name: str = "checkpoint",
    values_to_save: dict | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a checkpoint including model state_dict and optimizer state_dict to a file.

    Args:
        directory: Directory where the checkpoint will be saved.
        values_to_save: Additional values to save in the checkpoint.
        model: The PyTorch model to save.
        optimizer: The optimizer to save.
        checkpoint_name: The name of the checkpoint file.
    """
    from neps.runtime import get_in_progress_trial

    if directory is None:
        in_progress_trial = get_in_progress_trial()
        directory = in_progress_trial.metadata.location

    directory = Path(directory)
    checkpoint_path = (directory / checkpoint_name).with_suffix(".pth")

    saved_dict = {}

    if model is not None:
        saved_dict["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        saved_dict["optimizer_state_dict"] = optimizer.state_dict()

    if values_to_save is not None:
        saved_dict.update(values_to_save)

    torch.save(saved_dict, checkpoint_path)


def load_lightning_checkpoint(
    checkpoint_dir: Path | str,
    previous_pipeline_directory: Path | str | None = None,
) -> tuple[Path, dict] | tuple[None, None]:
    """Load the latest checkpoint file from the specified directory.

    This function searches for possible checkpoint files in the `checkpoint_dir` and loads
    the latest one if found. It returns a tuple with the checkpoint path and the loaded
    checkpoint data.

    Args:
        checkpoint_dir: The directory where checkpoint files are stored.
        previous_pipeline_directory: The previous pipeline directory.

    Returns:
        A tuple containing the checkpoint path (str) and the loaded checkpoint data (dict)
        or (None, None) if no checkpoint files are found in the directory.
    """
    from neps.runtime import get_in_progress_trial

    if previous_pipeline_directory is None:
        trial = get_in_progress_trial()
        previous_pipeline_directory = trial.metadata.previous_trial_location
        if previous_pipeline_directory is None:
            return None, None

    # Search for possible checkpoints to continue training
    ckpt_files = list(Path(checkpoint_dir).glob("*.ckpt"))

    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            "No checkpoint files were located in the checkpoint directory"
        )

    if len(ckpt_files) > 1:
        raise ValueError(
            "The number of checkpoint files is more than expected (1) "
            "which makes if difficult to find the correct file."
            " Please save other checkpoint files in a different directory."
        )

    assert len(ckpt_files) == 1
    checkpoint_path = ckpt_files[0]
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    return checkpoint_path, checkpoint


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
    from neps.runtime import get_in_progress_trial, get_workers_neps_state

    neps_state = get_workers_neps_state()
    if pipeline_directory is not None:
        # TODO: Hard coded assumption
        config_id = Path(pipeline_directory).name.split("_", maxsplit=1)[-1]
        trial = neps_state.unsafe_retry_get_trial_by_id(config_id)
    else:
        trial = get_in_progress_trial()

    if trial.metadata.id in _INTIAL_DIRECTORY_CACHE:
        return _INTIAL_DIRECTORY_CACHE[trial.metadata.id]

    # Recursively find the initial directory
    while (prev_trial_id := trial.metadata.previous_trial_id) is not None:
        trial = neps_state.unsafe_retry_get_trial_by_id(prev_trial_id)

    initial_dir = trial.metadata.location

    # TODO: Hard coded assumption that we are operating in a filebased neps
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


# TODO(eddiebergman): This seems like a bad function name, I guess this is used for a
# string somewhere.
def get_value(obj: Any) -> Any:
    """Honestly, don't know why you would use this. Please try not to."""
    if obj is None:
        return None
    if isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, dict):
        return {key: get_value(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [get_value(item) for item in obj]

    return obj.__name__


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
