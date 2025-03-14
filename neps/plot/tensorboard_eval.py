"""The tblogger module provides a simplified interface for logging to TensorBoard."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import override

import numpy as np
import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter

from neps.runtime import (
    get_in_progress_trial,
    get_workers_neps_state,
    register_notify_trial_end,
)
from neps.status.status import status
from neps.utils.common import get_initial_directory

if TYPE_CHECKING:
    from neps.state.trial import Trial


class SummaryWriter_(SummaryWriter):  # noqa: N801
    """This class inherits from the base SummaryWriter class and provides
    modifications to improve the logging. It simplifies the logging structure
    and ensures consistent tag formatting for metrics.

    Changes Made:
    - Avoids creating unnecessary subfolders in the log directory.
    - Ensures all logs are stored in the same 'tfevent' directory for
      better organization.
    - Updates metric keys to have a consistent 'Summary/' prefix for clarity.
    - Improves the display of 'objective_to_minimize' or 'Accuracy' on the Summary file.

    Methods:
    - add_hparams: Overrides the base method to log hyperparameters and
    metrics with better formatting.
    """

    @override
    def add_hparams(  # type: ignore
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, Any],
        global_step: int,
    ) -> None:
        """Add a set of hyperparameters to be logged."""
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        updated_metric = {f"Summary/{key}": val for key, val in metric_dict.items()}
        exp, ssi, sei = hparams(hparam_dict, updated_metric)

        assert self.file_writer is not None
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in updated_metric.items():
            self.add_scalar(tag=k, scalar_value=v, global_step=global_step)


class tblogger:  # noqa: N801
    """The tblogger class provides a simplified interface for logging to tensorboard."""

    config_id: ClassVar[str | None] = None
    config: ClassVar[Mapping[str, Any] | None] = None
    config_working_directory: ClassVar[Path | None] = None
    optimizer_dir: ClassVar[Path | None] = None
    config_previous_directory: ClassVar[Path | None] = None

    disable_logging: ClassVar[bool] = False

    objective_to_minimize: ClassVar[float | None] = None
    current_epoch: ClassVar[int | None] = None

    write_incumbent: ClassVar[bool | None] = None

    config_writer: ClassVar[SummaryWriter_ | None] = None
    summary_writer: ClassVar[SummaryWriter_ | None] = None

    @staticmethod
    def _initiate_internal_configurations() -> None:
        """Track the Configuration space data from the way handled by neps runtime
        '_sample_config' to keep in sync with config ids and directories NePS is
        operating on.
        """
        trial = get_in_progress_trial()
        neps_state = get_workers_neps_state()

        register_notify_trial_end("NEPS_TBLOGGER", tblogger.end_of_config)

        # We are assuming that neps state is all filebased here
        root_dir = Path(neps_state.path)
        assert root_dir.exists()

        tblogger.config_working_directory = Path(trial.metadata.location)
        tblogger.config_previous_directory = (
            Path(trial.metadata.previous_trial_location)
            if trial.metadata.previous_trial_location is not None
            else None
        )
        tblogger.config_id = trial.metadata.id
        tblogger.optimizer_dir = root_dir
        tblogger.config = trial.config

    @staticmethod
    def _is_initialized() -> bool:
        return tblogger.config_writer is not None

    @staticmethod
    def _initialize_writers() -> None:
        # This code runs only once per config, to assign that config a config_writer.
        if (
            tblogger.config_previous_directory is None
            and tblogger.config_working_directory is not None
        ):
            # If no fidelities are there yet, define the writer via the config_id
            tblogger.config_id = str(tblogger.config_working_directory).rsplit(
                "/", maxsplit=1
            )[-1]
            tblogger.config_writer = SummaryWriter_(
                tblogger.config_working_directory / "tbevents"
            )
            return

        # Searching for the initial directory where tensorboard events are stored.
        if tblogger.config_working_directory is not None:
            init_dir = get_initial_directory(
                pipeline_directory=tblogger.config_working_directory
            )
            tblogger.config_id = str(init_dir).rsplit("/", maxsplit=1)[-1]
            if (init_dir / "tbevents").exists():
                tblogger.config_writer = SummaryWriter_(init_dir / "tbevents")
                return

            raise FileNotFoundError(
                "'tbevents' was not found in the initial directory of the configuration."
            )

    @staticmethod
    def end_of_config(trial: Trial) -> None:  # noqa: ARG004
        """Closes the writer."""
        if tblogger.config_writer:
            # Close and reset previous config writers for consistent logging.
            # Prevent conflicts by reinitializing writers when logging ongoing.
            tblogger.config_writer.close()
            tblogger.config_writer = None

        if tblogger.write_incumbent:
            tblogger._tracking_incumbent_api()

    @staticmethod
    def _make_grid(images: torch.Tensor, nrow: int, padding: int = 2) -> torch.Tensor:
        """Create a grid of images from a batch of images.

        Args:
            images (torch.Tensor): The input batch of images with shape
                (batch_size, num_channels, height, width).
            nrow (int): The number of rows on the grid.
            padding (int, optional): The padding between images in the grid.
                Default is 2.

        Returns:
            torch.Tensor: A grid of images with shape:
                (num_channels, total_height, total_width)
        """
        batch_size, num_channels, height, width = images.size()
        x_mapping = min(nrow, batch_size)
        y_mapping = int(math.ceil(float(batch_size) / x_mapping))
        height, width = height + 2, width + 2

        grid = torch.zeros(
            (num_channels, height * y_mapping + padding, width * x_mapping + padding)
        )

        k = 0
        for y in range(y_mapping):
            for x in range(x_mapping):
                if k >= batch_size:
                    break
                image = images[k]
                grid[
                    :,
                    y * height + padding : y * height + padding + height - padding,
                    x * width + padding : x * width + padding + width - padding,
                ] = image
                k += 1

        return grid

    @staticmethod
    def scalar_logging(value: float) -> tuple[str, float]:
        """Prepare a scalar value for logging.

        Args:
            value (float): The scalar value to be logged.

        Returns:
            Tuple: A tuple containing the logging mode and the value for logging.
                The tuple format is (logging_mode, value).
        """
        logging_mode = "scalar"
        return (logging_mode, value)

    @staticmethod
    def image_logging(
        image: torch.Tensor,
        counter: int = 1,
        *,
        resize_images: list[None | int] | None = None,
        random_images: bool = True,
        num_images: int = 20,
        seed: int | np.random.RandomState | None = None,
    ) -> tuple[
        str,
        torch.Tensor,
        int,
        list[None | int] | None,
        bool,
        int,
        int | np.random.RandomState | None,
    ]:
        """Prepare an image tensor for logging.

        Args:
            image: Image tensor to be logged.
            counter: Counter value associated with the images.
            resize_images: List of integers for image sizes after resizing.
            random_images: Images are randomly selected if True.
            num_images: Number of images to log.
            seed: Seed value or RandomState instance to control randomness.

        Returns:
            A tuple containing the logging mode and all the necessary parameters for
            image logging.
            Tuple: (logging_mode, img_tensor, counter, resize_images,
                            random_images, num_images, seed).
        """
        logging_mode = "image"
        return (
            logging_mode,
            image,
            counter,
            resize_images,
            random_images,
            num_images,
            seed,
        )

    @staticmethod
    def _write_scalar_config(tag: str, value: float | int) -> None:
        """Write scalar values to the TensorBoard log.

        Args:
            tag (str): The tag for the scalar value.
            value (float or int): The scalar value to be logged. Default is None.

        Note:
            The function relies on the _initialize_writers to ensure the
            TensorBoard writer is initialized at the correct directory.

            It also depends on the following global variables:
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_id (str)

            The function will log the scalar value under different tags based
            on fidelity mode and other configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        # Just an extra safety measure
        if tblogger.config_writer is not None:
            tblogger.config_writer.add_scalar(
                tag=str(tblogger.config_id) + "/" + tag,
                scalar_value=value,
                global_step=tblogger.current_epoch,
            )
        else:
            raise ValueError(
                "The 'config_writer' is None in _write_scalar_config."
                "An error occurred during the initialization process."
            )

    @staticmethod
    def _write_image_config(
        tag: str,
        image: torch.Tensor,
        counter: int = 1,
        *,
        resize_images: list[None | int] | None = None,
        random_images: bool = True,
        num_images: int = 20,
        seed: int | np.random.RandomState | None = None,
    ) -> None:
        """Write images to the TensorBoard log.

        Note:
            The function relies on the _initialize_writers to ensure the
            TensorBoard writer is initialized at the correct directory.

            It also depends on the following global variables:
                - tblogger.current_epoch (int)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_id (str)

            The function will log a subset of images to TensorBoard based on
            the given configurations.

        Args:
            tag: Tag for the images.
            image: Image tensor to be logged.
            counter: Counter value associated with the images.
            resize_images: List of integers for image sizes after resizing.
            random_images: Images are randomly selected if True.
            num_images: Number of images to log.
            seed: Seed value or RandomState instance to control randomness.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        if resize_images is None:
            resize_images = [32, 32]

        assert tblogger.current_epoch is not None

        if tblogger.current_epoch >= 0 and tblogger.current_epoch % counter == 0:
            # Log every multiple of "counter"

            num_images = min(num_images, len(image))

            if random_images is False:
                subset_images = image[:num_images]
            else:
                if not isinstance(seed, np.random.RandomState):
                    seed = np.random.RandomState(seed)

                # We do not interfere with any randomness from the pipeline
                num_total_images = len(image)
                indices = seed.choice(num_total_images, num_images, replace=False)
                subset_images = image[indices]

            resized_images = torch.nn.functional.interpolate(
                subset_images,
                size=(resize_images[0], resize_images[1]),
                mode="bilinear",
                align_corners=False,
            )
            # Create the grid according to the number of images and log
            # the grid to tensorboard.
            nrow = int(resized_images.size(0) ** 0.75)
            img_grid = tblogger._make_grid(resized_images, nrow=nrow)
            # Just an extra safety measure
            if tblogger.config_writer is not None:
                tblogger.config_writer.add_image(
                    tag=str(tblogger.config_id) + "/" + tag,
                    img_tensor=img_grid,
                    global_step=tblogger.current_epoch,
                )
            else:
                raise ValueError(
                    "The 'config_writer' is None in _write_image_config."
                    "An error occurred during the initialization process."
                )

    @staticmethod
    def _write_hparam_config() -> None:
        """Write hyperparameter configurations to the TensorBoard log, inspired
        by the 'hparam' original function of tensorboard.

        Note:
            The function relies on the _initialize_writers to ensure the
            TensorBoard writer is initialized at the correct directory.

            It also depends on the following global variables:
                - tblogger.objective_to_minimize (float)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config (dict)
                - tblogger.current_epoch (int)

            The function will log hyperparameter configurations along
            with a metric value (either accuracy or objective_to_minimize) to TensorBoard
            based on the given configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        str_name = "Objective to minimize"
        str_value = tblogger.objective_to_minimize

        values = {str_name: str_value}
        # Just an extra safety measure
        if tblogger.config_writer is not None:
            assert tblogger.config is not None
            assert tblogger.current_epoch is not None

            tblogger.config_writer.add_hparams(
                hparam_dict=dict(tblogger.config),
                metric_dict=values,
                global_step=tblogger.current_epoch,
            )
        else:
            raise ValueError(
                "The 'config_writer' is None in _write_hparam_config."
                "An error occurred during the initialization process."
            )

    @staticmethod
    def _tracking_incumbent_api() -> None:
        """Track the incumbent (best) objective_to_minimize and log it in the TensorBoard
            summary.

        Note:
            The function relies on the following global variables:
                - tblogger.optimizer_dir
                - tblogger.summary_writer

            The function logs the incumbent trajectory in TensorBoard.
        """
        assert tblogger.optimizer_dir is not None
        _, short = status(tblogger.optimizer_dir, print_summary=False)

        incum_tracker = short["state.completed"]
        incum_val = short["best_objective_to_minimize"]

        if tblogger.summary_writer is None and tblogger.optimizer_dir is not None:
            tblogger.summary_writer = SummaryWriter_(tblogger.optimizer_dir / "summary")

        assert tblogger.summary_writer is not None
        tblogger.summary_writer.add_scalar(
            tag="Summary/Incumbent_graph",
            scalar_value=incum_val,
            global_step=incum_tracker,
        )

        # Frequent writer open/close creates new 'tfevent' files due to
        # parallelization needs. Simultaneous open writers risk conflicts,
        # so they're flushed and closed after use.

        tblogger.summary_writer.flush()
        tblogger.summary_writer.close()

    @staticmethod
    def disable() -> None:
        """The function allows for disabling the logger functionality.
        When the logger is disabled, it will not perform logging operations.

        By default tblogger is enabled when used.

        Example:
            # Disable the logger
            tblogger.disable()
        """
        tblogger.disable_logging = True

    @staticmethod
    def enable() -> None:
        """The function allows for enabling the logger functionality.
        When the logger is enabled, it will perform the logging operations.

        By default this is enabled.

        Example:
            # Enable the logger
            tblogger.enable()
        """
        tblogger.disable_logging = False

    @staticmethod
    def get_status() -> bool:
        """Returns the currect state of tblogger ie. whether the logger is
        enabled or not.
        """
        return not tblogger.disable_logging

    @staticmethod
    def log(
        objective_to_minimize: float,
        current_epoch: int,
        *,
        writer_config_scalar: bool = True,
        writer_config_hparam: bool = True,
        write_summary_incumbent: bool = False,
        extra_data: dict | None = None,
    ) -> None:
        """Log experiment data to the logger, including scalar values,
        hyperparameters, and images.

        Args:
            objective_to_minimize: Current objective_to_minimize value.
            current_epoch: Current epoch of the experiment (used as the global step).
            writer_config_scalar: Displaying the objective_to_minimize or accuracy
                curve on tensorboard (default: True)
            writer_config_hparam: Write hyperparameters logging of the configs.
            write_summary_incumbent: Set to `True` for a live incumbent trajectory.
            extra_data: Additional experiment data for logging.
        """
        if tblogger.disable_logging:
            return

        tblogger.current_epoch = current_epoch
        tblogger.objective_to_minimize = objective_to_minimize
        tblogger.write_incumbent = write_summary_incumbent

        tblogger._initiate_internal_configurations()

        if writer_config_scalar:
            tblogger._write_scalar_config(
                tag="objective_to_minimize", value=objective_to_minimize
            )

        if writer_config_hparam:
            tblogger._write_hparam_config()

        if extra_data is not None:
            for key in extra_data:
                if extra_data[key][0] == "scalar":
                    tblogger._write_scalar_config(tag=str(key), value=extra_data[key][1])

                elif extra_data[key][0] == "image":
                    tblogger._write_image_config(
                        tag=str(key),
                        image=extra_data[key][1],
                        counter=extra_data[key][2],
                        resize_images=extra_data[key][3],
                        random_images=extra_data[key][4],
                        num_images=extra_data[key][5],
                        seed=extra_data[key][6],
                    )
