import math
import os
import random
import warnings
from typing import List, Optional, Union

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

    def add_hparams(self, hparam_dict, metric_dict, global_step):
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        updated_metric = {}
        for key, value in metric_dict.items():
            updated_key = "Summary" + "/" + key
            updated_metric[updated_key] = value
        exp, ssi, sei = hparams(hparam_dict, updated_metric)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in updated_metric.items():
            self.add_scalar(tag=k, scalar_value=v, global_step=global_step)


class tblogger:
    config = None
    config_id: Optional[int] = None
    config_working_directory = None
    config_previous_directory = None
    optim_path = None

    config_value_fid: Optional[str] = None
    fidelity_mode: bool = False

    logger_init_bool: bool = True
    logger_bool: bool = False

    image_logger: bool = False
    image_value: Optional[torch.tensor] = None
    image_name: Optional[str] = None
    epoch_value: Optional[int] = None

    disable_logging: bool = False

    loss: Optional[float] = None
    current_epoch: int
    scalar_accuracy_mode: bool = False
    hparam_accuracy_mode: bool = False

    config_writer: Optional[SummaryWriter_] = None
    summary_writer: Optional[SummaryWriter_] = None

    logging_mode: list = []

    @staticmethod
    def config_track_init_api(
        config_id, config, config_working_directory, config_previous_directory, optim_path
    ):
        """
        Track the Configuration space data from the way handled by neps metahyper '_sample_config' to keep in sync with
        config ids and directories NePS is operating on.
        """

        tblogger.config = config
        tblogger.config_id = config_id
        tblogger.config_working_directory = config_working_directory
        tblogger.config_previous_directory = config_previous_directory
        tblogger.optim_path = optim_path

    @staticmethod
    def _initialize_writers():
        if not tblogger.config_writer:
            # If the writer is still not assgined
            optim_config_path = tblogger.optim_path / "results"
            if tblogger.config_previous_directory is not None:
                # If a previous directory is available (Now the search is done for higher fidelity but logging is
                # saved on the previous directory)
                tblogger.fidelity_mode = True
                while not tblogger.config_writer:
                    if os.path.exists(tblogger.config_previous_directory / "tbevents"):
                        # If the previous directory was actually the first fidelity,
                        # tbevents is the folder holding the logging event files "tfevent"
                        find_previous_config_id = (
                            tblogger.config_working_directory / "previous_config.id"
                        )
                        if os.path.exists(find_previous_config_id):
                            # Get the ID of the previous config to log on the new train data
                            with open(find_previous_config_id) as file:
                                contents = file.read()
                                tblogger.config_value_fid = contents
                                tblogger.config_writer = SummaryWriter_(
                                    tblogger.config_previous_directory / "tbevents"
                                )
                    else:
                        # If the directory does not have the writer created,
                        # find the previous config and keep on looping backward until locating
                        # the inital config holding the tfevent files
                        find_previous_config_path = (
                            tblogger.config_previous_directory / "previous_config.id"
                        )
                        if os.path.exists(find_previous_config_path):
                            with open(find_previous_config_path) as file:
                                contents = file.read()
                                tblogger.config_value_fid = contents
                                tblogger.config_working_directory = (
                                    tblogger.config_previous_directory
                                )
                                tblogger.config_previous_directory = (
                                    optim_config_path / f"config_{contents}"
                                )
            else:
                # If no fidelities are there, define the writer via the normal config_id
                tblogger.fidelity_mode = False
                tblogger.config_writer = SummaryWriter_(
                    tblogger.config_working_directory / "tbevents"
                )

    @staticmethod
    def _make_grid(images: torch.tensor, nrow: int, padding: int = 2):
        """
        Create a grid of images from a batch of images.

        Args:
            images (torch.Tensor): The input batch of images with shape (batch_size, num_channels, height, width).
            nrow (int): The number rows on the grid.
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
    def scalar_logging(value: float) -> list:
        """
        Prepare a scalar value for logging.

        Args:
            value (float): The scalar value to be logged.

        Returns:
            list: A list containing the logging mode and the value for logging.
                The list format is [logging_mode, value].
        """
        logging_mode = "scalar"
        return [logging_mode, value]

    @staticmethod
    def image_logging(
        img_tensor: torch.Tensor,
        counter: int,
        resize_images: Optional[List[Optional[int]]] = None,
        ignore_warning: bool = True,
        random_images: bool = True,
        num_images: int = 20,
    ) -> List[Union[str, torch.Tensor, int, bool, List[Optional[int]]]]:
        """
        Prepare an image tensor for logging.

        Args:
            img_tensor (torch.Tensor): The image tensor to be logged.
            counter (int): A counter value for teh frequency of image logging (ex: counter 2 means for every
                        2 epochs a new set of images are logged).
            resize_images (list of int): A list of integers representing the image sizes
                                                        after resizing or None if no resizing required.
                                                        Default is None.
            ignore_warning (bool, optional): Whether to ignore any warning during logging. Default is True.
            random_images (bool, optional): Whether the images are selected randomly. Default is True.
            num_images (int, optional): The number of images to log. Default is 20.

        Returns:
            list: A list containing the logging mode and all the necessary parameters for image logging.
                The list format is [logging_mode, img_tensor, counter, repetitive, resize_images,
                                    ignore_warning, random_images, num_images].
        """
        logging_mode = "image"
        return [
            logging_mode,
            img_tensor,
            counter,
            resize_images,
            ignore_warning,
            random_images,
            num_images,
        ]

    @staticmethod
    def _write_scalar_config(tag: str, value: Union[float, int]):
        """
        Write scalar values to the TensorBoard log.

        Args:
            tag (str): The tag for the scalar value.
            value (float or int): The scalar value to be logged. Default is None.

        Note:
            If the tag is 'Loss' and scalar_accuracy_mode is True, the tag will be changed to 'Accuracy',
            and the value will be transformed accordingly.

            The function relies on the initialize_config_writer to ensure the TensorBoard writer is initialized at
            the correct directory.

            It also depends on the following global variables:
                - tblogger.scalar_accuracy_mode (bool)
                - tblogger.fidelity_mode (bool)
                - tblogger.config_writer (SummaryWriter_)

            The function will log the scalar value under different tags based on fidelity mode and other configurations.
        """
        tblogger._initialize_writers()

        if tag == "Loss":
            if tblogger.scalar_accuracy_mode:
                tag = "Accuracy"
                value = (1 - value) * 100
        if tblogger.config_writer is not None:
            if tblogger.fidelity_mode:
                tblogger.config_writer.add_scalar(
                    tag="Config_" + str(tblogger.config_value_fid) + "/" + tag,
                    scalar_value=value,
                    global_step=tblogger.current_epoch,
                )
            else:
                tblogger.config_writer.add_scalar(
                    tag="Config_" + str(tblogger.config_id) + "/" + tag,
                    scalar_value=value,
                    global_step=tblogger.current_epoch,
                )

    @staticmethod
    def _write_image_config(
        tag: str,
        image: torch.tensor,
        counter: int,
        resize_images: Optional[List[Optional[int]]] = None,
        ignore_warning: bool = True,
        random_images: bool = True,
        num_images: int = 20,
    ):
        """
        Write images to the TensorBoard log.

        Args:
            tag (str): The tag for the images.
            image (torch.Tensor): The image tensor to be logged.
            counter (int): A counter value associated with the images.
            resize_images (list of int): A list of integers representing the image sizes
                                                        after resizing or None if no resizing required.
                                                        Default is None.
            ignore_warning (bool, optional): Whether to ignore any warning during logging. Default is True.
            random_images (bool, optional): Whether the images are selected randomly. Default is True.
            num_images (int, optional): The number of images to log. Default is 20.

        Note:
            The function relies on the initialize_config_writer to ensure the TensorBoard writer is initialized at
            the correct directory.

            It also depends on the following global variables:
                - tblogger.current_epoch (int)
                - tblogger.fidelity_mode (bool)
                - tblogger.config_writer (SummaryWriter_)
                - tblogger.config_value_fid (int or None)
                - tblogger.config_id (int)

            The function will log a subset of images to TensorBoard based on the given configurations.
        """
        tblogger._initialize_writers()

        if resize_images is None:
            resize_images = [32, 32]

        if ignore_warning is True:
            warnings.filterwarnings("ignore", category=DeprecationWarning)

        if tblogger.current_epoch % counter == 0:
            # Log every multiple of  "counter"

            if num_images > len(image):
                # Be safe if the number of images is not as the len (as in the batch size)
                num_images = len(image)

            if random_images is False:
                subset_images = image[:num_images]
            else:
                random_indices = random.sample(range(len(image)), num_images)
                subset_images = image[random_indices]

            resized_images = torch.nn.functional.interpolate(
                subset_images,
                size=(resize_images[0], resize_images[1]),
                mode="bilinear",
                align_corners=False,
            )
            # Create the grid according to the number of images and log the grid to tensorboard.
            nrow = int(resized_images.size(0) ** 0.75)
            img_grid = tblogger._make_grid(resized_images, nrow=nrow)
            if tblogger.config_writer is not None:
                if tblogger.fidelity_mode:
                    tblogger.config_writer.add_image(
                        tag="Config_" + str(tblogger.config_value_fid) + "/" + tag,
                        img_tensor=img_grid,
                        global_step=tblogger.current_epoch,
                    )
                else:
                    tblogger.config_writer.add_image(
                        tag="Config_" + str(tblogger.config_id) + "/" + tag,
                        img_tensor=img_grid,
                        global_step=tblogger.current_epoch,
                    )

    @staticmethod
    def _write_hparam_config():
        """
        Write hyperparameter configurations to the TensorBoard log, inspired by the 'hparam' original function of tensorboard.

        Note:
            The function relies on the initialize_config_writer to ensure the TensorBoard writer is initialized at
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
        tblogger._initialize_writers()

        if tblogger.hparam_accuracy_mode:
            # Changes the loss to accuracy and logs in accuracy terms.
            str_name = "Accuracy"
            str_value = (1 - tblogger.loss) * 100
        else:
            str_name = "Loss"
            str_value = tblogger.loss

        values = {str_name: str_value}
        if tblogger.config_writer is not None:
            tblogger.config_writer.add_hparams(
                hparam_dict=tblogger.config,
                metric_dict=values,
                global_step=tblogger.current_epoch,
            )

    @staticmethod
    def tracking_incumbent_api(best_loss):
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
            # Close all the previous config writers
            tblogger.config_writer.close()
            tblogger.config_writer = None

        file_path = str(tblogger.optim_path) + "/all_losses_and_configs.txt"
        tblogger.incum_tracker = 0
        with open(file_path) as f:
            for line in f:
                # Count the amount of presence of "Config ID" because it correlates to the
                # step size of how many configurations were completed.
                tblogger.incum_tracker += line.count("Config ID")

        tblogger.incum_val = float(best_loss)

        logdir = str(tblogger.optim_path) + "/summary"

        if tblogger.summary_writer is None:
            tblogger.summary_writer = SummaryWriter_(logdir)

        tblogger.summary_writer.add_scalar(
            tag="Summary" + "/Incumbent_graph",
            scalar_value=tblogger.incum_val,
            global_step=tblogger.incum_tracker,
        )

        tblogger.summary_writer.flush()
        tblogger.summary_writer.close()

    @staticmethod
    def disable(disable_logger: bool = True):
        """
        The function allows for enabling or disabling the logger functionality
        throughout the program execution by updating the value of 'tblogger.disable_logging'.
        When the logger is disabled, it will not perform any logging operations.

        Args:
            disable_logger (bool, optional): A boolean flag to control the logger.
                                            If True (default), the logger will be disabled.
                                            If False, the logger will be enabled.

        Example:
            # Disable the logger
            tblogger.disable()

            # Enable the logger
            tblogger.disable(False)
        """
        tblogger.disable_logging = disable_logger

    @staticmethod
    def log(
        loss: float,
        current_epoch: int,
        writer_scalar: bool = True,
        writer_hparam: bool = True,
        scalar_accuracy_mode: bool = False,
        hparam_accuracy_mode: bool = False,
        data: Optional[dict] = None,
    ):
        """
        Log experiment data to the logger, including scalar values, hyperparameters, images, and layer gradients.

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
                                    'tag2': tblogger.image_logging(img_tensor=img, counter=2),
                                    'tag3': tblogger.layer_gradient_logging(model=model),
                                }
                                Default is None.

        """
        tblogger.current_epoch = current_epoch
        tblogger.loss = loss
        tblogger.scalar_accuracy_mode = scalar_accuracy_mode
        tblogger.hparam_accuracy_mode = hparam_accuracy_mode

        if not tblogger.disable_logging:
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
                            ignore_warning=data[key][4],
                            random_images=data[key][5],
                            num_images=data[key][6],
                        )

        else:
            tblogger.logger_bool = False
