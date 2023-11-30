from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from ..metahyper.api import ConfigInRun
from ..status.status import get_summary_dict
from ..utils.common import get_initial_directory


class SummaryWriter_(SummaryWriter):
    """
    This class inherits from the base SummaryWriter class and provides
    modifications to improve the logging. It simplifies the logging structure
    and ensures consistent tag formatting for metrics.

    Changes Made:
    - Avoids creating unnecessary subfolders in the log directory.
    - Ensures all logs are stored in the same 'tfevent' directory for
      better organization.
    - Updates metric keys to have a consistent 'Summary/' prefix for clarity.
    - Improves the display of 'Loss' or 'Accuracy' on the Summary file.

    Methods:
    - add_hparams: Overrides the base method to log hyperparameters and
    metrics with better formatting.
    """

    def add_hparams(self, hparam_dict: dict, metric_dict: dict, global_step: int) -> None:
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        updated_metric = {f"Summary/{key}": val for key, val in metric_dict.items()}
        exp, ssi, sei = hparams(hparam_dict, updated_metric)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in updated_metric.items():
            self.add_scalar(tag=k, scalar_value=v, global_step=global_step)


class tblogger:
    config_id: str | None = None
    config: dict | None = None
    config_working_directory: Path | None = None
    optim_path: Path | None = None
    config_previous_directory: Path | None = None

    disable_logging: bool = False

    logger_bool: bool = False
    """logger_bool is true only if tblogger.log is used by the user, this
    allows to always capturing the configuration data."""

    loss: float | None = None
    current_epoch: int | None = None

    write_incumbent: bool | None = None

    config_writer: SummaryWriter_ | None = None
    summary_writer: SummaryWriter_ | None = None

    @staticmethod
    def _initiate_internal_configurations() -> None:
        """
        Track the Configuration space data from the way handled by neps metahyper
        '_sample_config' to keep in sync with config ids and directories NePS is
        operating on.
        """
        tblogger.config_working_directory = ConfigInRun.pipeline_directory
        tblogger.config_previous_directory = ConfigInRun.previous_pipeline_directory
        tblogger.optim_path = ConfigInRun.optimization_dir
        tblogger.config = ConfigInRun.config

    @staticmethod
    def _is_initialized() -> bool:
        # Returns 'True' if config_writer is already initialized. 'False' otherwise
        return tblogger.config_writer is not None

    @staticmethod
    def _initialize_writers() -> None:
        # This code runs only once per config, to assign that config a config_writer.
        if (
            tblogger.config_previous_directory is None
            and tblogger.config_working_directory
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
        if tblogger.config_working_directory:
            init_dir = get_initial_directory(
                pipeline_directory=tblogger.config_working_directory
            )
            tblogger.config_id = str(init_dir).rsplit("/", maxsplit=1)[-1]
            if os.path.exists(init_dir / "tbevents"):
                tblogger.config_writer = SummaryWriter_(init_dir / "tbevents")
                return
            else:
                raise FileNotFoundError(
                    "'tbevents' was not found in the initial directory of the configuration."
                )

    @staticmethod
    def end_of_config():
        if tblogger.config_writer:
            # Close and reset previous config writers for consistent logging.
            # Prevent conflicts by reinitializing writers when logging ongoing.
            tblogger.config_writer.close()
            tblogger.config_writer = None

        if tblogger.write_incumbent:
            tblogger._tracking_incumbent_api()

    @staticmethod
    def _make_grid(images: torch.Tensor, nrow: int, padding: int = 2) -> torch.Tensor:
        """
        Create a grid of images from a batch of images.

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
        """
        Prepare a scalar value for logging.

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
        """
        Prepare an image tensor for logging.

        Args:
            image (torch.Tensor): Image tensor to be logged.
            counter (int): Counter value associated with the images.
            resize_images (list of int, optional): List of integers for image
                sizes after resizing (default: None).
            random_images (bool, optional): Images are randomly selected
                if True (default: True).
            num_images (int, optional): Number of images to log (default: 20).
            seed (int or np.random.RandomState or None, optional): Seed value
                or RandomState instance to control randomness (default: None).

        Returns:
            Tuple: A tuple containing the logging mode and all the necessary
            parameters for image logging.
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
        """
        Write scalar values to the TensorBoard log.

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
        resize_images: list[None | int] | None = None,
        random_images: bool = True,
        num_images: int = 20,
        seed: int | np.random.RandomState | None = None,
    ) -> None:
        """
        Write images to the TensorBoard log.

        Args:
            tag (str): Tag for the images.
            image (torch.Tensor): Image tensor to be logged.
            counter (int): Counter value associated with the images.
            resize_images (list of int, optional): List of integers for image
                sizes after resizing (default: None).
            random_images (bool, optional): Images are randomly selected
                if True (default: True).
            num_images (int, optional): Number of images to log (default: 20).
            seed (int or np.random.RandomState or None, optional): Seed value
                or RandomState instance to control randomness (default: None).

        Note:
            The function relies on the _initialize_writers to ensure the
            TensorBoard writer is initialized at the correct directory.

            It also depends on the following global variables:
                - tblogger.current_epoch (int)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_id (str)

            The function will log a subset of images to TensorBoard based on
            the given configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        if resize_images is None:
            resize_images = [32, 32]

        if tblogger.current_epoch >= 0 and tblogger.current_epoch % counter == 0:
            # Log every multiple of "counter"

            if num_images > len(image):
                # If the number of images requested by the user
                # is more than the ones available.
                num_images = len(image)

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
        """
        Write hyperparameter configurations to the TensorBoard log, inspired
        by the 'hparam' original function of tensorboard.

        Note:
            The function relies on the _initialize_writers to ensure the
            TensorBoard writer is initialized at the correct directory.

            It also depends on the following global variables:
                - tblogger.loss (float)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config (dict)
                - tblogger.current_epoch (int)

            The function will log hyperparameter configurations along
            with a metric value (either accuracy or loss) to TensorBoard
            based on the given configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        str_name = "Loss"
        str_value = tblogger.loss

        values = {str_name: str_value}
        # Just an extra safety measure
        if tblogger.config_writer is not None:
            tblogger.config_writer.add_hparams(
                hparam_dict=tblogger.config,
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
        """
        Track the incumbent (best) loss and log it in the TensorBoard summary.

        Note:
            The function relies on the following global variables:
                - tblogger.optim_path (str)
                - tblogger.summary_writer (SummaryWriter_)

            The function logs the incumbent trajectory in TensorBoard.
        """
        summary_dict = get_summary_dict(tblogger.optim_path, add_details=True)

        incum_tracker = summary_dict["num_evaluated_configs"]
        incum_val = summary_dict["best_loss"]

        if tblogger.summary_writer is None and tblogger.optim_path:
            tblogger.summary_writer = SummaryWriter_(tblogger.optim_path / "summary")

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
        """
        The function allows for disabling the logger functionality.
        When the logger is disabled, it will not perform logging operations.

        By default tblogger is enabled when used.

        Example:
            # Disable the logger
            tblogger.disable()
        """
        tblogger.disable_logging = True

    @staticmethod
    def enable() -> None:
        """
        The function allows for enabling the logger functionality.
        When the logger is enabled, it will perform the logging operations.

        By default this is enabled.

        Example:
            # Enable the logger
            tblogger.enable()
        """
        tblogger.disable_logging = False

    @staticmethod
    def get_status():
        """
        Returns the currect state of tblogger ie. whether the logger is
        enabled or not
        """
        return not tblogger.disable_logging

    @staticmethod
    def log(
        loss: float,
        current_epoch: int,
        writer_config_scalar: bool = True,
        writer_config_hparam: bool = True,
        write_summary_incumbent: bool = False,
        extra_data: dict | None = None,
    ) -> None:
        """
        Log experiment data to the logger, including scalar values,
        hyperparameters, and images.

        Args:
            loss (float): Current loss value.
            current_epoch (int): Current epoch of the experiment
                (used as the global step).
            writer_scalar (bool, optional): Displaying the loss or accuracy
                curve on tensorboard (default: True)
            writer_hparam (bool, optional): Write hyperparameters logging of
                the configs (default: True).
            extra_data (dict, optional): Additional experiment data for logging.
        """
        if tblogger.disable_logging:
            tblogger.logger_bool = False
            return

        tblogger.logger_bool = True

        tblogger.current_epoch = current_epoch
        tblogger.loss = loss
        tblogger.write_incumbent = write_summary_incumbent

        tblogger._initiate_internal_configurations()

        if writer_config_scalar:
            tblogger._write_scalar_config(tag="Loss", value=loss)

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
