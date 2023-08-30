from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class SummaryWriter_(SummaryWriter):
    """
    This class inherits from the base SummaryWriter class and provides modifications to improve the logging.
    It simplifies the logging structure and ensures consistent tag formatting for metrics.

    Changes Made:
    - Avoids creating unnecessary subfolders in the log directory.
    - Ensures all logs are stored in the same 'tfevent' directory for better organization.
    - Updates metric keys to have a consistent 'Summary/' prefix for clarity.
    - Improves the display of 'Loss' or 'Accuracy' on the Summary file.

    Methods:
    - add_hparams: Overrides the base method to log hyperparameters and metrics with better formatting.
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

    logger_init_bool: bool = True
    """logger_init_bool is only used once to capture configuration data for the first ever configuration,
    and then turned false for the entire run."""

    logger_bool: bool = False
    """logger_bool is true only if tblogger.log is used by the user, hence this allows to always capturing
    the configuration data for all configurations."""

    disable_logging: bool = False
    """disable_logging is a hard switch to disable the logging feature if it was turned true.
    hence even when logger_bool is true it disables the logging process"""

    loss: float | None = None
    current_epoch: int | None = None
    scalar_accuracy_mode: bool = False
    hparam_accuracy_mode: bool = False

    incum_tracker: int | None = None
    incum_val: float | None = None

    config_writer: SummaryWriter_ | None = None
    summary_writer: SummaryWriter_ | None = None

    @staticmethod
    def config_track_init_api(
        config_id: str,
        config: dict,
        config_working_directory: Path,
        optim_path: Path,
        config_previous_directory: Path | None = None,
    ) -> None:
        """
        Track the Configuration space data from the way handled by neps metahyper '_sample_config' to keep in sync with
        config ids and directories NePS is operating on.
        """

        tblogger.config = config
        tblogger.config_id = config_id
        tblogger.config_working_directory = config_working_directory
        tblogger.optim_path = optim_path
        tblogger.config_previous_directory = config_previous_directory

    @staticmethod
    def _is_initialized() -> bool:
        # Returns 'True' if config_writer is already initialized. 'False' otherwise
        return tblogger.config_writer is not None

    @staticmethod
    def _initialize_writers() -> None:
        # This code runs only once per config, to assign that config a config_writer.
        if tblogger.config_previous_directory is None:
            # If no fidelities are there yet, define the writer via the config_id
            tblogger.config_writer = SummaryWriter_(
                tblogger.config_working_directory / "tbevents"
            )
            return
        while not tblogger._is_initialized():
            # Ensure proper directory for TensorBoard data appending.
            # Search for the first fidelity directory to store tfevent files.

            prev_dir_id_from_init = (
                tblogger.config_working_directory / "previous_config.id"
            )
            if tblogger.config_previous_directory is not None:
                get_tbevent_dir = tblogger.config_previous_directory / "tbevents"
            else:
                warnings.warn(
                    "There should be a previous config directory at this stage."
                    "Prone to failure"
                )

            # This should execute when having Config_x_1 and Config_x_0
            if os.path.exists(get_tbevent_dir):
                # When tfevents directory is detected => we are at the first fidelity directory, create writer.
                with open(prev_dir_id_from_init) as file:
                    contents = file.read()
                    tblogger.config_id = contents
                tblogger.config_writer = SummaryWriter_(
                    tblogger.config_previous_directory / "tbevents"
                )
                return

            # This should execute when having Config_x_y and Config_x_y where y > 0
            if tblogger.config_previous_directory is not None:
                prev_dir_id_from_prev = (
                    tblogger.config_previous_directory / "previous_config.id"
                )
            else:
                warnings.warn(
                    "There should be a previous config directory at this stage."
                    "Prone to failure"
                )

            if os.path.exists(prev_dir_id_from_prev):
                # To get the new previous config directory
                with open(prev_dir_id_from_prev) as file:
                    contents = file.read()
                    tblogger.config_id = contents
                    tblogger.config_working_directory = tblogger.config_previous_directory
                    tblogger.config_previous_directory = (
                        tblogger.optim_path / "results" / f"config_{contents}"
                    )
            else:
                # If no tbevents found after traversing all config directories,
                # raise an error to prevent indefinite 'while 1' loop.
                raise FileNotFoundError(
                    "'tbevents' was not found in the directory of the initial fidelity."
                )

    @staticmethod
    def _make_grid(images: torch.Tensor, nrow: int, padding: int = 2) -> torch.Tensor:
        """
        Create a grid of images from a batch of images.

        Args:
            images (torch.Tensor): The input batch of images with shape (batch_size, num_channels, height, width).
            nrow (int): The number of rows on the grid.
            padding (int, optional): The padding between images in the grid. Default is 2.

        Returns:
            torch.Tensor: A grid of images with shape (num_channels, total_height, total_width),
                        where total_height and total_width depend on the number of images and the grid settings.
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
    def scalar_logging(value: float) -> tuple:
        """
        Prepare a scalar value for logging.

        Args:
            value (float): The scalar value to be logged.

        Returns:
            tuple: A tuple containing the logging mode and the value for logging.
                The tuple format is (logging_mode, value).
        """
        logging_mode = "scalar"
        return (logging_mode, value)

    @staticmethod
    def image_logging(
        img_tensor: torch.Tensor,
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
            img_tensor (torch.Tensor): The image tensor to be logged.
            counter (int): A counter value for the frequency of image logging (ex: counter 2 means for every
                        2 global steps a new set of images are logged).
            resize_images (list of int): A list of integers representing the image sizes
                                                    after resizing or None if no resizing required.
                                                    Default is None.
            random_images (bool, optional): Whether the images are selected randomly. Default is True.
            num_images (int, optional): The number of images to log. Default is 20.
            seed (int or np.random.RandomState or None, optional): Seed value or RandomState instance to control
                                                               the randomness of image selection. Default is None.

        Returns:
            tuple: A tuple containing the logging mode and all the necessary parameters for image logging.
                The tuple format is (logging_mode, img_tensor, counter, resize_images,
                                    random_images, num_images, seed).
        """
        logging_mode = "image"
        return (
            logging_mode,
            img_tensor,
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
            If the tag is 'Loss' and scalar_accuracy_mode is True, the tag will be changed to 'Accuracy',
            and the value will be transformed accordingly.

            The function relies on the _initialize_writers to ensure the TensorBoard writer is initialized at
            the correct directory.

            It also depends on the following global variables:
                - tblogger.scalar_accuracy_mode (bool)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_id (str)

            The function will log the scalar value under different tags based on fidelity mode and other configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        if tag == "Loss" and tblogger.scalar_accuracy_mode:
            tag = "Accuracy"
            value = (1 - value) * 100

        # Just an extra safety measure
        if tblogger.config_writer is not None:
            tblogger.config_writer.add_scalar(
                tag="Config_" + str(tblogger.config_id) + "/" + tag,
                scalar_value=value,
                global_step=tblogger.current_epoch,
            )
        else:
            raise ValueError(
                "The 'config_writer' is None in _write_scalar_config. No logging is performed."
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
            tag (str): The tag for the images.
            image (torch.Tensor): The image tensor to be logged.
            counter (int): A counter value associated with the images.
            resize_images (list of int): A list of integers representing the image sizes
                                                        after resizing or None if no resizing required.
                                                        Default is None.
            random_images (bool, optional): Whether the images are selected randomly. Default is True.
            num_images (int, optional): The number of images to log. Default is 20.
            seed (int or np.random.RandomState or None, optional): Seed value or RandomState instance to control
                                                               the randomness of image selection. Default is None.

        Note:
            The function relies on the _initialize_writers to ensure the TensorBoard writer is initialized at
            the correct directory.

            It also depends on the following global variables:
                - tblogger.current_epoch (int)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_id (str)

            The function will log a subset of images to TensorBoard based on the given configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        if resize_images is None:
            resize_images = [32, 32]

        if tblogger.current_epoch % counter == 0:
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
            # Create the grid according to the number of images and log the grid to tensorboard.
            nrow = int(resized_images.size(0) ** 0.75)
            img_grid = tblogger._make_grid(resized_images, nrow=nrow)
            # Just an extra safety measure
            if tblogger.config_writer is not None:
                tblogger.config_writer.add_image(
                    tag="Config_" + str(tblogger.config_id) + "/" + tag,
                    img_tensor=img_grid,
                    global_step=tblogger.current_epoch,
                )
            else:
                raise ValueError(
                    "The 'config_writer' is None in _write_image_config. No logging is performed. "
                    "An error occurred during the initialization process."
                )

    @staticmethod
    def _write_hparam_config() -> None:
        """
        Write hyperparameter configurations to the TensorBoard log, inspired by the 'hparam' original function of tensorboard.

        Note:
            The function relies on the _initialize_writers to ensure the TensorBoard writer is initialized at
            the correct directory.

            It also depends on the following global variables:
                - tblogger.hparam_accuracy_mode (bool)
                - tblogger.loss (float)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config (dict)
                - tblogger.current_epoch (int)

            The function will log hyperparameter configurations along with a metric value (either accuracy or loss)
            to TensorBoard based on the given configurations.
        """
        if not tblogger._is_initialized():
            tblogger._initialize_writers()

        if tblogger.hparam_accuracy_mode:
            # Changes the loss to accuracy and logs in accuracy terms.
            str_name = "Accuracy"
            str_value = (1 - tblogger.loss) * 100
        else:
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
                "The 'config_writer' is None in _write_hparam_config. No logging is performed. "
                "An error occurred during the initialization process."
            )

    @staticmethod
    def tracking_incumbent_api(best_loss: float) -> None:
        """
        Track the incumbent (best) loss and log it in the TensorBoard summary.

        Args:
            best_loss (float): The best loss value to be tracked, according to the _post_hook_function of NePS.

        Note:
            The function relies on the following global variables:
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.optim_path (str)
                - tblogger.incum_tracker (int)
                - tblogger.incum_val (float)
                - tblogger.summary_writer (SummaryWriter_)

            The function logs the incumbent loss in a TensorBoard summary with a graph.
            It increments the incumbent tracker based on occurrences of "Config ID" in the 'all_losses_and_configs.txt' file.
        """
        if tblogger.config_writer:
            # Close and reset previous config writers for consistent logging.
            # Prevent conflicts by reinitializing writers when logging ongoing.
            tblogger.config_writer.close()
            tblogger.config_writer = None

        file_path = str(tblogger.optim_path) + "/all_losses_and_configs.txt"
        tblogger.incum_tracker = 0
        if os.path.exists(file_path):
            with open(file_path) as f:
                # Count "Config ID" occurrences to track completed configurations.
                tblogger.incum_tracker = sum(line.count("Config ID") for line in f)
        else:
            raise FileExistsError(
                "all_losses_and_configs.txt does not exist in the optimization directory"
            )

        tblogger.incum_val = float(best_loss)

        logdir = str(tblogger.optim_path) + "/summary"

        if tblogger.summary_writer is None:
            tblogger.summary_writer = SummaryWriter_(logdir)

        tblogger.summary_writer.add_scalar(
            tag="Summary/Incumbent_graph",
            scalar_value=tblogger.incum_val,
            global_step=tblogger.incum_tracker,
        )

        # Frequent writer open/close creates new 'tfevent' files due to parallelization needs.
        # Simultaneous open writers risk conflicts, so they're flushed and closed after use.

        tblogger.summary_writer.flush()
        tblogger.summary_writer.close()

    @staticmethod
    def disable() -> None:
        """
        The function allows for disabling the logger functionality
        throughout the program execution by updating the value of 'tblogger.disable_logging'.
        When the logger is disabled, it will not perform any logging operations.

        By default tblogger is enabled when used. If for any reason disabling is needed. This function does the job.

        Example:
            # Disable the logger
            tblogger.disable()
        """
        tblogger.disable_logging = True

    @staticmethod
    def enable() -> None:
        """
        The function allows for enabling the logger functionality
        throughout the program execution by updating the value of 'tblogger.disable_logging'.
        When the logger is enabled, it will perform the logging operations.

        By default this is enabled. Hence only needed when tblogger was once disabled.

        Example:
            # Enable the logger
            tblogger.enable()
        """
        tblogger.disable_logging = False

    @staticmethod
    def get_status():
        """
        Returns the currect state of tblogger ie. whether the logger is enabled or not
        """
        return not tblogger.disable_logging

    @staticmethod
    def log(
        loss: float,
        current_epoch: int,
        writer_scalar: bool = True,
        writer_hparam: bool = True,
        scalar_accuracy_mode: bool = False,
        hparam_accuracy_mode: bool = False,
        data: dict | None = None,
    ) -> None:
        """
        Log experiment data to the logger, including scalar values, hyperparameters, and images.

        Args:
            loss (float): The current loss value in training.
            current_epoch (int): The current epoch of the experiment. Used as the global step.
            writer_scalar (bool, optional): Whether to write the loss or accuracy for the
                                        configs during training. Default is True.
            writer_hparam (bool, optional): Whether to write hyperparameters logging
                                        of the configs during training. Default is True.
            scalar_accuracy_mode (bool, optional): If True, interpret the 'loss' as 'accuracy' and transform it's
                                                value accordingliy. Default is False.
            hparam_accuracy_mode (bool, optional): If True, interpret the 'loss' as 'accuracy' and transform it's
                                                value accordingliy. Default is False.
            data (dict, optional): Additional experiment data to be logged. It should be in the format:
                                {
                                    'tag1': tblogger.scalar_logging(value=value1),
                                    'tag2': tblogger.image_logging(img_tensor=img, counter=2, seed=0),
                                }
                                Default is None.

        """
        tblogger.current_epoch = current_epoch
        tblogger.loss = loss
        tblogger.scalar_accuracy_mode = scalar_accuracy_mode
        tblogger.hparam_accuracy_mode = hparam_accuracy_mode

        if tblogger.disable_logging:
            tblogger.logger_bool = False
            return

        tblogger.logger_bool = True

        if writer_scalar:
            tblogger._write_scalar_config(tag="Loss", value=loss)

        if writer_hparam:
            tblogger._write_hparam_config()

        if data is not None:
            for key in data:
                if data[key][0] == "scalar":
                    tblogger._write_scalar_config(tag=str(key), value=data[key][1])

                elif data[key][0] == "image":
                    tblogger._write_image_config(
                        tag=str(key),
                        image=data[key][1],
                        counter=data[key][2],
                        resize_images=data[key][3],
                        random_images=data[key][4],
                        num_images=data[key][5],
                        seed=data[key][6],
                    )
