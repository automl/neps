from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..metahyper.api import ConfigInRun
from ..optimizers.info import SearcherConfigs


def load_checkpoint(
    directory: Path | str | None = None,
    checkpoint_name: str = "checkpoint",
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict | None:
    """
    Load a checkpoint and return the model state_dict and checkpoint values.

    Args:
        directory (Path or str, optional): Directory where the checkpoint is located.
        checkpoint_name (str, optional): The name of the checkpoint file. Default
            is "checkpoint.pth".
        model (torch.nn.Module, optional): The PyTorch model to load. Default is
            None.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load. Default
            is None.

    Returns:
        Union[dict, None]: A dictionary containing the checkpoint values, or None if
        the checkpoint file does not exist hence no checkpointing was previously done.
    """
    # Check if the user did not provide a specific pipeline directory
    # or if the provided pipeline directory does not exist.
    if directory is None:
        # If not provided, use the pipeline directory from ConfigInRun
        directory = ConfigInRun.previous_pipeline_directory

    # If the pipeline directory remains None even in ConfigInRun, return None.
    # Otherwise, create a Path object using the provided or ConfigInRun value.
    if directory:
        directory = Path(directory)

    checkpoint_path = f"{directory}/{checkpoint_name}.pth"

    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path)

    # Load the model's state_dict and optimizer's state_dict if provided
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from '{checkpoint_path}'")

    # Return the checkpoint values
    return checkpoint


def save_checkpoint(
    directory: Path | str | None = None,
    checkpoint_name: str = "checkpoint",
    values_to_save: dict | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    """
    Save a checkpoint including model state_dict and optimizer state_dict to a file.

    Args:
        directory (Path or str): Directory where the checkpoint will be saved.
        values_to_save (dict, optional): Additional values to save in the checkpoint.
            Default is None.
        model (torch.nn.Module, optional): The PyTorch model to save. Default is
            None.
        optimizer (torch.optim.Optimizer, optional): The optimizer to save. Default
            is None.
        checkpoint_name (str, optional): The name of the checkpoint file. Default
            is "checkpoint.pth".
    """
    if directory is None:
        directory = ConfigInRun.pipeline_directory

    directory = Path(directory)
    checkpoint_path = f"{directory}/{checkpoint_name}.pth"

    saved_dict = {}

    # Add model state_dict and optimizer state_dict if provided
    if model is not None:
        saved_dict["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        saved_dict["optimizer_state_dict"] = optimizer.state_dict()

    # Update saved_dict with additional values if provided
    if values_to_save is not None:
        saved_dict.update(values_to_save)

    # Save the checkpoint to the specified path
    torch.save(saved_dict, checkpoint_path)


def load_lightning_checkpoint(
    checkpoint_dir: Path | str, previous_pipeline_directory: Path | str | None = None
) -> tuple[str | None, dict | None]:
    """
    Load the latest checkpoint file from the specified directory.

    This function searches for possible checkpoint files in the `checkpoint_dir` and loads
    the latest one if found. It returns a tuple with the checkpoint path and the loaded
    checkpoint data.

    Args:
        previous_pipeline_directory (Union[Path, str, None]): The previous pipeline directory.
        checkpoint_dir (Union[Path, str]): The directory where checkpoint files are stored.

    Returns:
        Tuple[Optional[str], Optional[dict]]: A tuple containing the checkpoint path (str)
        and the loaded checkpoint data (dict). Returns (None, None) if no checkpoint files
        are found in the directory.
    """
    if previous_pipeline_directory is None:
        previous_pipeline_directory = ConfigInRun.previous_pipeline_directory

    if previous_pipeline_directory:
        # Search for possible checkpoints to continue training
        ckpt_files = glob.glob(str(Path(checkpoint_dir) / "*.ckpt"))

        if len(ckpt_files) > 1:
            raise ValueError(
                "The number of checkpoint files is more than expected (1) "
                "which makes if difficult to find the correct file."
                " Please save other checkpoint files in a different directory."
            )

        if ckpt_files:
            # Load the checkpoint and retrieve necessary data
            checkpoint_path = ckpt_files[-1]
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(
                "No checkpoint files were located in the checkpoint directory"
            )
        return checkpoint_path, checkpoint
    else:
        return None, None


def get_initial_directory(pipeline_directory: Path | str | None = None) -> Path:
    """
    Find the initial directory based on its existence and the presence of
    the "previous_config.id" file.

    Args:
        pipeline_directory (Union[Path, str, None]): The current config directory.

    Returns:
        Path: The initial directory.
    """
    if pipeline_directory is None:
        pipeline_directory = ConfigInRun.pipeline_directory

    pipeline_directory = Path(pipeline_directory)

    while True:
        # Get the id of the previous directory
        previous_pipeline_directory_id = pipeline_directory / "previous_config.id"

        # Get the directory where all configs are saved
        optim_result_dir = pipeline_directory.parent

        if previous_pipeline_directory_id.exists():
            # Get and join to the previous path according to the id
            with open(previous_pipeline_directory_id) as config_id_file:
                config_id = config_id_file.read()
                pipeline_directory = optim_result_dir / f"config_{config_id}"
        else:
            # Initial directory found
            return pipeline_directory


def get_searcher_data(searcher: str, searcher_path: Path | str | None = None) -> dict:
    """
    Returns the data from the YAML file associated with the specified searcher.

    Args:
        searcher (str): The name of the searcher.
        searcher_path (Path | None, optional): The path to the directory where
            the searcher defined YAML file is located. Defaults to None.

    Returns:
        dict: The content of the YAML file.
    """

    if searcher_path:
        user_yaml_path = os.path.join(Path(searcher_path), f"{searcher}.yaml")

        if not os.path.exists(user_yaml_path):
            raise FileNotFoundError(
                f"File '{searcher}.yaml' does not exist in {os.getcwd()}."
            )

        with open(user_yaml_path) as file:
            data = yaml.safe_load(file)

    else:
        folder_path = "optimizers/default_searchers"
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.join(script_directory, os.pardir)
        resource_path = os.path.join(parent_directory, folder_path, f"{searcher}.yaml")

        searchers = SearcherConfigs.get_searchers()

        if not os.path.exists(resource_path):
            raise FileNotFoundError(
                f"Searcher '{searcher}' not in:\n{', '.join(searchers)}"
            )

        with open(resource_path) as file:
            data = yaml.safe_load(file)

    return data


def get_value(obj: Any):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {key: get_value(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [get_value(item) for item in obj]
    else:
        return obj.__name__


def has_instance(collection, *types):
    return any([isinstance(el, typ) for el in collection for typ in types])


def filter_instances(collection, *types):
    return [el for el in collection if any([isinstance(el, typ) for typ in types])]


def get_rnd_state() -> dict:
    np_state = list(np.random.get_state())
    np_state[1] = np_state[1].tolist()
    state = {
        "random_state": random.getstate(),
        "np_seed_state": np_state,
        "torch_seed_state": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_seed_state"] = [
            dev.tolist() for dev in torch.cuda.get_rng_state_all()
        ]
    return state


def set_rnd_state(state: dict):
    # rnd_s1, rnd_s2, rnd_s3 = state["random_state"]
    random.setstate(
        tuple(
            tuple(rnd_s) if isinstance(rnd_s, list) else rnd_s
            for rnd_s in state["random_state"]
        )
    )
    np.random.set_state(tuple(state["np_seed_state"]))
    torch.random.set_rng_state(torch.ByteTensor(state["torch_seed_state"]))
    if torch.cuda.is_available() and "torch_cuda_seed_state" in state:
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(dev) for dev in state["torch_cuda_seed_state"]]
        )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class DataWriter:
    """
    A class to specify how to save/write a data to the folder by
     implementing your own write_data function.
     Use the set_attributes function to set all your necessary attributes and the data
     and then write_data will be called with only the directory path as argument
     during the write process
    """
    def __init__(self, name: str):
        self.name = name

    def set_attributes(self, attribute_dict: dict[str, Any]):
        for attribute_name, attribute in attribute_dict.items():
            setattr(self, attribute_name, attribute)

    def write_data(self, to_directory: Path):
        raise NotImplementedError


class EvaluationData:
    """
    A class to store some data for a single evaluation (configuration)
     and write that data to its corresponding config folder
    """
    def __init__(self):
        self.data_dict: dict[str, DataWriter] = {}

    def write_all(self, directory: Path):
        for _, data_writer in self.data_dict.items():
            data_writer.write_data(directory)


class SimpleCSVWriter(DataWriter):
    def write_data(self, to_directory: Path):
        # self.df: pd.DataFrame = pd.DataFrame()
        path = to_directory / str(self.name + ".csv")
        self.df.to_csv(path, float_format="%g")


class SimpleJSONWriter(DataWriter):
    def __init__(self):
        self.data: dict[str, Any] = {}

    def write_data(self, to_directory: Path):
        # self.df: pd.DataFrame = pd.DataFrame()
        path = to_directory / str(self.name + ".json")
        with open(path, "w") as file:
            json.dump(self.data, file)







