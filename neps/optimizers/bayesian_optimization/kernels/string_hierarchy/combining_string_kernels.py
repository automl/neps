from __future__ import annotations

import abc
import logging

import torch

from . import config_string

_logger = logging.getLogger(__name__)


# Convert configs from SearchSpace into StringConfig objects

def get_config_hps_by_category(configs: tuple) -> tuple[dict[str, list]]:
    """ Returns a tuple of dicts.
    For each config, return its hyperparameters by category
    Categories like: "continuous", "categorical", "string", "graphs"
    """
    config_hps = tuple(conf.get_normalized_hp_categories() for conf in configs)
    return config_hps


def get_config_string_objects(config_strings) -> tuple[config_string.ConfigString]:
    """Return the object representation of the list of string configs"""
    # Ensure that each config has one and only one such hp
    for config_string_hps in config_strings:
        if len(config_string_hps) != 1:
            raise ValueError(
                "Found config with multiple string hyperparameters." +
                f"config string hyperparameters={config_string_hps!r}"
            )
    # Take the (string) value of the first config
    config_strings = [config_string_hps[0] for config_string_hps in config_strings]
    # Construct a collection of representation objects for the string configs
    config_strings = tuple(
        config_string.ConfigString(config_string=c)
        for c in config_strings
    )
    return config_strings


# Define the combining kernel

class CombiningKernel(abc.ABC):
    def __init__(
        self,
        combining_kernel_variant: str,
        kernels: tuple,
    ):
        if combining_kernel_variant not in ("sum", "product"):
            raise ValueError(
                "Invalid value for combining kernel variant: "
                + f"{combining_kernel_variant!r}"
            )
        self.combining_kernel_variant = combining_kernel_variant
        self.kernels = tuple(kernels)

    def transform(
        self,
        weights: torch.Tensor,
        configs: tuple,
        normalize: bool = True,
    ):
        n_configs = len(configs)
        n_kernels = len(self.kernels)

        if (weights < 0).any():
            raise ValueError(f"Found values <0 in weights: {weights}")
        elif (torch.sum(weights) - 1).item() > 1e-5:
            weight_sum = torch.sum(weights).item()
            raise ValueError(f"Weights don't sum up to 1: {weights}, {weight_sum}")

        # For each config, get its hyperparameters by category
        config_hps_by_category = get_config_hps_by_category(configs=configs)
        # For each config, take only the string hyperparameters
        #  and convert them into a wrapper object with more information
        config_strings = get_config_string_objects(
            config_strings=[hps["string"] for hps in config_hps_by_category],
        )

        if not config_strings:
            raise ValueError(
                f"No configurations found: config_strings={config_strings!r}"
            )
        elif len(config_strings) != n_configs:
            raise ValueError(
                "Wrong number of config_strings (len(config_strings) != n_configs): "
                + f"{len(config_strings)} != {n_configs}"
            )

        # Compute the result for each kernel
        kernel_results: list[torch.Tensor] = []
        for kernel_index, kernel in enumerate(self.kernels):
            K_i = kernel.transform(configs=config_strings)
            if normalize:
                K_i_diag = torch.sqrt(torch.diag(K_i))
                K_i = K_i / torch.ger(K_i_diag, K_i_diag)
            kernel_results.append(K_i)

        # Stack the kernel results along
        kernel_results_tensor = torch.stack(kernel_results, dim=0)
        assert kernel_results_tensor.shape == (n_kernels, n_configs, n_configs)

        # Multiply kernel results by their weights
        kernel_results_tensor = weights.view(-1, 1, 1) * kernel_results_tensor
        assert kernel_results_tensor.shape == (n_kernels, n_configs, n_configs)

        # Do the combining operation
        if self.combining_kernel_variant == "sum":
            K = torch.sum(kernel_results_tensor, dim=0)
        elif self.combining_kernel_variant == "product":
            K = torch.prod(kernel_results_tensor, dim=0)
        else:
            raise ValueError(
                "Invalid value for combining kernel variant: "
                + f"{self.combining_kernel_variant!r}"
            )
        assert K.shape == (n_configs, n_configs)

        return K
