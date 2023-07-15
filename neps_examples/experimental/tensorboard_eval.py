import math
import random
import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


# Inherit from class and change to fit purpose:
class SummaryWriter_(SummaryWriter):
    """
    This function before the update used to create another subfolder inside the logdir and then create further 'tfevent'
    which makes everything else uneasy to differentiate and hence this gives the same result with a much easier way and logs
    everything on the same 'tfevent' as for other functions.
    In addition, a change in the metric dictiornay was made for the cause of making the printed 'Loss' or 'Accuracy' display on the
    Summary file
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


class tensorboard_evaluations:
    def __init__(self, log_dir: str = "/logs") -> None:
        self._log_dir = log_dir

        self._best_incum_track = np.inf
        self._step_update = 1

        self._toggle_epoch_max_reached = False

        self._config_track = 1

        self._fidelity_search_count = 0
        self._fidelity_counter = 0
        self._fidelity_bool = False
        self._fidelity_was_bool = False

        self._config_dict: dict[str, dict[str, Union[list[str], float, int]]] = {}
        self._config_track_last = 1
        self._prev_config_list: list[str] = []

        self._writer_config = []
        self._writer_summary = SummaryWriter_(log_dir=self._log_dir + "/summary")
        self._writer_config.append(
            SummaryWriter_(
                log_dir=self._log_dir + "/configs" + "/config_" + str(self._config_track)
            )
        )

    def _make_grid(self, images: torch.tensor, nrow: int, padding: int = 2):
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

    def _incumbent(self, **incum_data) -> None:
        """
        A function used to mainly display out the incumbent trajectory based on the step update which is after finishing every computation.
        In other words, epochs == max_epochs
        """
        loss = incum_data["loss"]
        if loss < self._best_incum_track:
            self._best_incum_track = loss
        self._writer_summary.add_scalar(
            tag="Summary" + "/Incumbent_Graph",
            scalar_value=self._best_incum_track,
            global_step=self._step_update,
        )
        self._step_update += 1

    def _track_config(self, **config_data) -> None:
        config_list = config_data["config_list"]
        loss = float(config_data["loss"])

        for config_dict in self._config_dict.values():
            if self._prev_config_list != config_list:
                if config_dict["config_list"] == config_list:
                    if self._fidelity_search_count == 0:
                        self._config_track_last = self._config_track
                        self._fidelity_was_bool = True
                    self._fidelity_bool = True
                    self._fidelity_search_count += 1
                    loss_prev = self._config_dict["config_" + str(self._config_track)][
                        "loss"
                    ]
                    self._incumbent(loss=loss_prev)
                    config = config_dict["config"]
                    if isinstance(config, (int, float)):
                        self._config_track = int(config)

        if not self._fidelity_bool:
            if len(self._prev_config_list) > 0:
                if self._prev_config_list != config_list:
                    self._fidelity_search_count = 0
                    if self._fidelity_was_bool:
                        loss_prev = self._config_dict[
                            "config_" + str(self._config_track)
                        ]["loss"]
                        self._incumbent(loss=loss_prev)
                        self._config_track = self._config_track_last + 1
                        self._fidelity_counter += 1
                        self._config_dict.clear()
                        self._fidelity_was_bool = False
                    else:
                        loss_prev = self._config_dict[
                            "config_" + str(self._config_track)
                        ]["loss"]
                        self._incumbent(loss=loss_prev)
                        self._config_track += 1
                    self._writer_config.append(
                        SummaryWriter_(
                            log_dir=self._log_dir
                            + "/configs"
                            + "/config_"
                            + str(self._config_track)
                        )
                    )
        else:
            self._fidelity_bool = False
            self._toggle_epoch_max_reached = False

        self._config_dict["config_" + str(self._config_track)] = {
            "config_list": config_list,
            "loss": float(loss),
            "config": self._config_track,
        }

        self._prev_config_list = config_list

    def write_scalar_configs(
        self, config_list: list, current_epoch: int, loss: float, scalar: float, tag: str
    ) -> None:
        """
        Writes any scalar to the specific corresponding config, EX: Learning_rate decay tracking, Accuracy...

        Arguments:
            conifg_list:    a list (The configurations sved as a list in run_pipline and passed here as an argument)
            current_epoch:  an integer (The currecnt epoch running at the time)
            loss:           a float (The loss at the specific run, important for hypeband)
            scalar:         a float (The scalar value to be visualized)
            tag:            a string (The tag of the scalar EX: tag = 'Learning_Rate')
        """
        if tag == "loss":
            scalar = loss

        if loss is None or current_epoch is None or config_list is None:
            raise ValueError(
                "Loss, epochs, and max_epochs cannot be None. Please provide a valid value."
            )

        self._track_config(config_list=config_list, loss=loss)

        self._writer_config[self._config_track - 1].add_scalar(
            tag="Config" + str(self._config_track) + "/" + tag,
            scalar_value=scalar,
            global_step=current_epoch,
        )

    def write_scalar_fidelity(
        self, config_list: list, current_epoch: int, loss: float, Accuracy: bool = False
    ) -> None:
        """
        This function will take the each fidelity and show the accuracy or the loss during HPO search for each fidelity.

        Arguments:
            conifg_list:    a list (The configurations sved as a list in run_pipline and passed here as an argument)
            current_epoch:  an integer (The currecnt epoch running at the time)
            loss:           a float (The loss at the specific run, important for hypeband)
            Accuracy:       a bool (If true it will change the loss to accuracy % and display the results.
                                    If false it will remain displaying with respect to the loss)
        """
        if loss is None or current_epoch is None or config_list is None:
            raise ValueError(
                "Loss, epochs, and max_epochs cannot be None. Please provide a valid value."
            )

        self._track_config(config_list=config_list, loss=loss)

        if Accuracy:
            acc = (1 - loss) * 100
            scalar_value = acc
        else:
            scalar_value = loss

        self._writer_config[self._config_track - 1].add_scalar(
            tag="Summary" + "/Fidelity_" + str(self._fidelity_counter),
            scalar_value=scalar_value,
            global_step=current_epoch,
        )

    def write_histogram(
        self, config_list: list, current_epoch: int, loss: float, model: nn.Module
    ) -> None:
        """
        By logging histograms for all parameters, you can gain insights into the distribution of different
        parameter types and identify potential issues or patterns in their values. This comprehensive analysis
        can help you better understand your model's behavior during training.

        Ex: Weights where their histograms do not show a change in shape from the first epoch up until the last prove to
        mean that the training is not done properly and hence weights are not updated in the rythm they should

        Arguments:
            conifg_list:    a list (The configurations sved as a list in run_pipline and passed here as an argument)
            current_epoch:  an integer (The currecnt epoch running at the time)
            loss:           a float (The loss at the specific run, important for hypeband)
            model:          a nn.Module (The model which we want to analyze)
        """
        if loss is None or current_epoch is None or config_list is None:
            raise ValueError(
                "Loss, epochs, and max_epochs cannot be None. Please provide a valid value."
            )

        self._track_config(config_list=config_list, loss=loss)

        for _, param in model.named_parameters():
            self._writer_config[self._config_track - 1].add_histogram(
                "Config" + str(self._config_track),
                param.clone().cpu().data.numpy(),
                current_epoch,
            )

    def write_image(
        self,
        config_list: list,
        max_epochs: int,
        current_epoch: int,
        loss: float,
        image_input: torch.Tensor,
        num_images: int = 10,
        random_images: bool = False,
        resize_images: np.array = None,
        ignore_warning: bool = True,
    ) -> None:
        """
        The user is free on how they want to tackle image visualization on tensorboard, they specify the numebr of images
        they want to show and if the images should be taken randomly or not.

        Arguments:
            conifg_list:    a list (The configurations sved as a list in run_pipline and passed here as an argument)
            max_epochs:     an integer (Maximum epoch that can be reached at that specific run)
            current_epoch:  an integer (The currecnt epoch running at the time)
            loss:           a float (The loss at the specific run)
            image_imput:    a Tensor (The input image in batch, shape: 12x3x28x28 'BxCxWxH')
            num_images:     an integer (The number of images ot be displayed for each config on tensorboard)
            random_images:  a bool (True is the images should be sampled randomly, False otherwise)
            resize_images:  an array (Resizing an the images to make them fit and be clearly visible on the grid)
            ignore_warning: a bool (At the moment a warning is appearing, bug will be fixed later)

        Example code of displaying wrongly classified images:

        1- In the trianing for loop:
        predicted_labels = torch.argmax(output_of_model_after_input, dim=1)
        misclassification_mask = predicted_labels != y_actual_labels
        misclassified_images.append(x[misclassification_mask])

        2- Before the return, outside the training loop:
        if len(misclassified_images) > 0:
            misclassified_images = torch.cat(misclassified_images, dim=0)

        3- Returning the misclassified images
        return ..., misclassified_images

        Then use these misclassified_images as the image_input of this function
        """
        if loss is None or current_epoch is None or config_list is None:
            raise ValueError(
                "Loss, epochs, and max_epochs cannot be None. Please provide a valid value."
            )

        self._track_config(config_list=config_list, loss=loss)

        if resize_images is None:
            resize_images = [56, 56]

        if ignore_warning is True:
            warnings.filterwarnings("ignore", category=DeprecationWarning)

        if current_epoch == max_epochs - 1:
            if num_images > len(image_input):
                num_images = len(image_input)

            if random_images is False:
                subset_images = image_input[:num_images]
            else:
                random_indices = random.sample(range(len(image_input)), num_images)
                subset_images = image_input[random_indices]

            resized_images = torch.nn.functional.interpolate(
                subset_images,
                size=(resize_images[0], resize_images[1]),
                mode="bilinear",
                align_corners=False,
            )

            nrow = int(resized_images.size(0) ** 0.75)
            img_grid = self._make_grid(resized_images, nrow=nrow)

            self._writer_config[self._config_track - 1].add_image(
                tag="IMG_config " + str(self._config_track),
                img_tensor=img_grid,
                global_step=self._config_track,
            )

    def write_hparam(
        self,
        config_list: list,
        current_epoch: int,
        loss: float,
        Accuracy: bool = False,
        **pipeline_space,
    ) -> None:
        """
        '.add_hparam' is a function in TensorBoard that allows you to log hyperparameters associated with your training run.
        It takes a dictionary of hyperparameter names and values and associates them with the current run, making it easy to
        compare and analyze different hyperparameter configurations.

        Arguments:
            conifg_list:    a list (The configurations sved as a list in run_pipline and passed here as an argument)
            current_epoch:  an integer (The currecnt epoch running at the time)
            loss:           a float (The loss at the specific run)
            Accuracy:       a bool (If true it will change the loss to accuracy % and display the results.
                                    If false it will remain displaying with respect to the loss)
            pipeline_space: The name of the hyperparameters in addition to their kwargs to be searched on.
        """
        if loss is None or current_epoch is None or config_list is None:
            raise ValueError(
                "Loss, epochs, and max_epochs cannot be None. Please provide a valid value."
            )

        self._track_config(config_list=config_list, loss=loss)

        if Accuracy:
            str_name = "Accuracy"
            str_value = (1 - loss) * 100
        else:
            str_name = "Loss"
            str_value = loss

        values = {str_name: str_value}

        self._writer_config[self._config_track - 1].add_hparams(
            pipeline_space, values, current_epoch
        )

    def close_writers(self) -> None:
        """
        Closing the writers created after finishing all the tensorboard visualizations
        """
        self._writer_summary.close()
        for _, writer in enumerate(self._writer_config):
            writer.close()
