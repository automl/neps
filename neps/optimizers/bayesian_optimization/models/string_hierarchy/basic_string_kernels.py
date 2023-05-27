from __future__ import annotations

import copy
import logging
from typing import Iterator

import torch

from ...kernels.string_hierarchy import config_string
from ...kernels.string_hierarchy import basic_string_kernels
from .. import gp_hierarchy as graph_gp_hierarchy  # Needed for math helper functions

_logger = logging.getLogger(__name__)


# Convert configs from SearchSpace into StringConfig objects

def _get_config_hps_by_category(configs: tuple) -> tuple[dict[str, list]]:
    """ Returns a tuple of dicts.
    For each config, return its hyperparameters by category
    Categories like: "continuous", "categorical", "string", "graphs"
    """
    config_hps = tuple(conf.get_normalized_hp_categories() for conf in configs)
    return config_hps


def _get_config_string_objects(config_strings) -> tuple[config_string.ConfigString]:
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


def _get_config_strings(configs: tuple) -> tuple[config_string.ConfigString]:
    # For each config, get its hyperparameters by category
    config_hps_by_category = _get_config_hps_by_category(configs=configs)

    # For each config, take only the string hyperparameters
    #  and convert them into a wrapper object with more information
    config_strings = _get_config_string_objects(
        config_strings=[hps["string"] for hps in config_hps_by_category],
    )
    if not config_strings:
        raise ValueError(
            f"No configurations found: config_strings={config_strings!r}"
        )
    elif len(config_strings) != len(configs):
        raise ValueError(
            "Wrong number of config_strings (len(config_strings) != len(configs)): "
            + f"{len(config_strings)} != {len(configs)}"
        )

    return config_strings


# Architecture string kernel: ASK

class ASK:
    def __init__(
        self,
        hierarchy_level: int | None = None,
        learnable_weights: bool = True,
    ):
        self.hierarchy_level = hierarchy_level
        self.learnable_weights = learnable_weights

        # Reset the model state
        self._reset_state()

    def _reset_state(self):
        self._kernel = basic_string_kernels.ASK(
            hierarchy_level=self.hierarchy_level,
            learnable_weights=self.learnable_weights,
        )

    def _optimize_kernel(
        self,
        config_strings: tuple[config_string.ConfigString],
        train_y: torch.Tensor,
        iters: int,
        optimizer: str,
        optimizer_kwargs: dict | None,
        likelihood: float,
        gpytorch_kinv: bool,
    ) -> dict:
        # Likelihood will not be optimized
        likelihood = torch.tensor(likelihood)
        likelihood.requires_grad_(False)

        # Whether the parameters of the kernel are optimized
        optim_vars = []
        for var in self._kernel.parameters():
            if var.requires_grad:
                optim_vars.append(var)
        _logger.debug("Kernel vars to optimize: \n%r", optim_vars)

        if optim_vars:
            if iters <= 0:
                raise ValueError(f"Variables to optimize, but `iters`<=0: {iters}")

            if optimizer.lower() == "adam":
                optim = torch.optim.Adam(optim_vars, **optimizer_kwargs)
            elif optimizer.lower() == "sgd":
                optim = torch.optim.SGD(optim_vars, **optimizer_kwargs)
            else:
                raise ValueError(f"Unknown optimizer {optimizer!r}")

            # Keep track of the best seen values
            best_nlml_value = None
            best_kernel_state_dict = None

            # The 0th iteration uses the initial weights,
            #  the 1st iteration uses the weights after one optimization iteration,
            #  hence the `(iters + 1)` for `iters` optimization iterations
            for i in range(iters + 1):
                _logger.debug("Starting optimization iteration %d", i)

                # Adjust the weights to an appropriate range
                with torch.no_grad():
                    torch.clamp_(self._kernel.weights, min=1e-5)

                # Reset the gradients
                optim.zero_grad(set_to_none=True)

                # Forward pass
                self._kernel.train()
                K = self._kernel.forward(configs=config_strings)
                K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
                    K, likelihood, gpytorch_kinv=gpytorch_kinv
                )

                # Compute the loss
                nlml = -graph_gp_hierarchy.compute_log_marginal_likelihood(
                    K_i=K_i,
                    logDetK=logDetK,
                    y=train_y,
                )

                # Check if best seen results achieved and if so, store the values
                # Make no operations affecting gradients here!
                iter_nlml = nlml.item()
                if best_nlml_value is None or iter_nlml < best_nlml_value:
                    best_nlml_value = nlml.clone().detach()
                    best_kernel_state_dict = copy.deepcopy(self._kernel.state_dict())
                    _logger.debug("Optimization values improved in iteration %d", i)
                    _logger.debug(
                        "Iteration: %d/%d. (state_dict, likelihood, NLML): \n%r",
                        i, iters, (self._kernel.state_dict(), likelihood, nlml)
                    )

                # Do a backward pass on the loss
                nlml.backward()

                # Check that gradients were collected and make an optimization step
                for var in optim_vars:
                    assert var.is_leaf, var
                    assert var.grad is not None, var
                    _logger.debug("Optim var and gradient: %s :: %s", var, var.grad)
                optim.step()

            # Now that the optimization is finished,
            #  set the variables to the best seen values for them
            assert best_kernel_state_dict is not None
            assert best_nlml_value is not None
            kernel_state_dict = best_kernel_state_dict
        else:
            kernel_state_dict = self._kernel.state_dict()

        return kernel_state_dict

    def fit_transform(
        self,
        train_x: tuple,
        train_y: torch.Tensor,
        iters: int,
        optimizer: str,
        optimizer_kwargs: dict | None,
        likelihood: float,
        gpytorch_kinv: bool,
    ) -> torch.Tensor:
        train_y = train_y.clone().detach()
        config_strings = _get_config_strings(configs=train_x)

        self._reset_state()

        optimized_kernel_state_dict = self._optimize_kernel(
            config_strings=config_strings,
            train_y=train_y,
            iters=iters,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            likelihood=likelihood,
            gpytorch_kinv=gpytorch_kinv,
        )
        _logger.debug("Loading optimized kernel state: %s", optimized_kernel_state_dict)
        self._kernel.load_state_dict(state_dict=optimized_kernel_state_dict)

        K = self._transform(config_strings=config_strings)
        return K

    def _transform(
        self,
        config_strings: tuple[config_string.ConfigString],
    ) -> torch.Tensor:
        self._kernel.eval()
        K = self._kernel.forward(configs=config_strings)
        return K

    def transform(self, x_configs: tuple) -> torch.Tensor:
        config_strings = _get_config_strings(configs=x_configs)
        return self._transform(config_strings=config_strings)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return list(self._kernel.parameters())
