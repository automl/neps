"""Common utility functions used across the library."""

from __future__ import annotations

import gc
import inspect
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml


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


def get_searcher_data(
    searcher: str | Path, *, loading_custom_searcher: bool = False
) -> tuple[dict[str, Any], str]:
    """Returns the data from the YAML file associated with the specified searcher.

    Args:
        searcher: The name of the searcher.
        loading_custom_searcher: Flag if searcher contains a custom yaml

    Returns:
        The content of the YAML file and searcher name.
    """
    if loading_custom_searcher:
        user_yaml_path = Path(searcher).with_suffix(".yaml")

        if not user_yaml_path.exists():
            raise FileNotFoundError(
                "Failed to get info for searcher from user-defined YAML file. "
                f"File '{searcher}.yaml' does not exist at '{user_yaml_path}'"
            )

        with user_yaml_path.open("r") as file:
            data = yaml.safe_load(file)

        file_name = user_yaml_path.stem
        searcher = data.pop("name", file_name)

    else:
        # TODO(eddiebergman): This is a bad idea as it relies on folder structure to be
        # correct, we should either have a dedicated resource folder or at least have
        # this defined as a constant somewhere, incase we access elsewhere.
        # Seems like we could just include this as a method on `SearcherConfigs` class.
        # TODO(eddiebergman): Need to make sure that these yaml files are actually
        # included in a source dist when published to PyPI.

        # This is pointing to yaml file directory elsewhere in the source code.
        resource_path = (
            Path(__file__).parent.parent.absolute()
            / "optimizers"
            / "default_searchers"
            / searcher
        ).with_suffix(".yaml")

        from neps.optimizers.info import SearcherConfigs

        searchers = SearcherConfigs.get_searchers()

        if not resource_path.exists():
            raise FileNotFoundError(
                f"Searcher '{searcher}' not in:\n{', '.join(searchers)}"
            )

        with resource_path.open() as file:
            data = yaml.safe_load(file)

    return data, searcher  # type: ignore


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


def instance_from_map(  # noqa: C901
    mapping: dict[str, Any],
    request: str | list | tuple | type,
    name: str = "mapping",
    *,
    allow_any: bool = True,
    as_class: bool = False,
    kwargs: dict | None = None,
) -> Any:
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
    if isinstance(request, Sequence) and not isinstance(request, str):
        if len(request) != 2:
            raise ValueError(
                "When building an instance and specifying arguments, "
                "you should give a pair (class, arguments)"
            )
        request, req_args_dict = request

        if not isinstance(req_args_dict, Mapping):
            raise ValueError("The arguments should be given as a dictionary")

        args_dict = {**args_dict, **req_args_dict}

    # Then, get the class/instance from the request
    if isinstance(request, str):
        if request not in mapping:
            raise ValueError(f"{request} doesn't exists for {name}")

        instance = mapping[request]
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
        instance = partial(instance, **args_dict)  # type: ignore

    if as_class:
        return instance

    if is_partial_class(instance):
        try:
            instance = instance()  # type: ignore
        except TypeError as e:
            raise TypeError(f"{e} when calling {instance} with {args_dict}") from e

    return instance


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
