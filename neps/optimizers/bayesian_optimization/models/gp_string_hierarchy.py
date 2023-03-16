from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from ..kernels.string_hierarchy import config_string
from ..kernels.string_hierarchy import StringKernelV1, CombiningKernel
from . import gp_hierarchy as graph_gp_hierarchy  # Needed for some helper functions

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


class GPStringHierarchy:
    def __init__(
        self,
        graph_kernels: Iterable[StringKernelV1],
        hp_kernels: Iterable | None = None,
        likelihood: float = 1e-3,
        combining_kernel_variant: str = "sum",
        surrogate_model_fit_args: dict = None,
        gpytorch_kinv: bool = False,
        vectorial_features: int | None = None,
    ):
        if hp_kernels:
            raise ValueError("HP kernels are not supported!")
        if vectorial_features:
            raise ValueError("Vectorial features are not supported!")

        self.kernels = tuple(graph_kernels)
        self.combining_kernel_variant = combining_kernel_variant
        self.surrogate_model_fit_args = surrogate_model_fit_args or {}
        self.gpytorch_kinv = gpytorch_kinv
        self.init_likelihood = float(likelihood)

        # Reset the model state
        self._reset_model_state()

    def _reset_model_state(
        self,
        train_x: tuple | None = None,
        train_y: tuple | None = None,
    ):
        self.x_configs = train_x
        if train_y is not None:
            train_y = (
                train_y
                if isinstance(train_y, torch.Tensor)
                else torch.tensor(train_y, dtype=torch.get_default_dtype())
            )
            self.y, self.y_mean, self.y_std = graph_gp_hierarchy.normalize_y(train_y)
            self.y_ = train_y  # used upstream
        else:
            self.y, self.y_mean, self.y_std = None, None, None
            self.y_ = None  # used upstream

        self.combining_kernel = CombiningKernel(
            n_kernels=len(self.kernels),
            combining_kernel_variant=self.combining_kernel_variant,
        )

        self.likelihood: float | None = self.init_likelihood
        self.nlml: float | None = None
        self.K: torch.Tensor | None = None
        self.K_i: torch.Tensor | None = None
        self.logDetK: torch.Tensor | None = None

    @property
    def x(self):  # used upstream
        return self.x_configs

    def _get_config_strings(
        self,
        configs: tuple,
    ) -> tuple[config_string.ConfigString]:
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
        elif len(config_strings) != len(configs):
            raise ValueError(
                "Wrong number of config_strings (len(config_strings) != len(configs)): "
                + f"{len(config_strings)} != {len(configs)}"
            )

        return config_strings

    def fit(self, train_x: Iterable, train_y: Iterable):
        train_x = tuple(train_x)
        train_y = tuple(train_y)
        self._reset_model_state(train_x=train_x, train_y=train_y)
        self._fit(**self.surrogate_model_fit_args)

    def _optimize_kernel(
        self,
        kernel: StringKernelV1,
        configs: tuple[config_string.ConfigString],
        iters: int,
        optimizer: str,
        optimizer_kwargs: dict | None,
        likelihood: float,
    ):
        likelihood = torch.tensor(likelihood)

        optim_vars = []
        for var in kernel.parameters():
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
            best_seen_kernel_state_dict = None
            lowest_seen_nlml_value = None

            # The 0th iteration uses the initial weights,
            #  the 1st iteration uses the weights after one optimization iteration,
            #  hence the `(iters + 1)` for `iters` optimization iterations
            for i in range(iters + 1):
                _logger.debug("Starting optimization iteration %d", i)

                # Reset the gradients
                optim.zero_grad(set_to_none=True)

                # Forward pass
                K = kernel.forward(configs=configs)
                K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
                    K, likelihood, gpytorch_kinv=self.gpytorch_kinv
                )

                # Compute the loss
                nlml = -graph_gp_hierarchy.compute_log_marginal_likelihood(
                    K_i=K_i,
                    logDetK=logDetK,
                    y=self.y,
                )

                # Check if best seen results achieved and if so, store the values
                # Ensure no operations affecting gradients are done here
                iter_nlml = nlml.item()
                if lowest_seen_nlml_value is None or iter_nlml < lowest_seen_nlml_value:
                    lowest_seen_nlml_value = nlml.clone().detach()
                    best_seen_kernel_state_dict = kernel.state_dict()
                    _logger.debug("Optimization values improved in iteration %d", i)

                if i % 10 == 0:
                    _logger.debug(
                        "Iteration: %d/%d. (state_dict, likelihood, NLML): \n%r",
                        i, iters, (kernel.state_dict(), likelihood, nlml)
                    )

                # Do a backward pass on the loss and then an optimization step
                nlml.backward()

                for var in optim_vars:
                    assert var.is_leaf, var
                    assert var.grad is not None, var
                    _logger.debug("Optim var and gradient: %s :: %s", var, var.grad)
                optim.step()

            # Now that the optimization is finished,
            #  set the variables to the best seen values for them
            assert best_seen_kernel_state_dict is not None
            assert lowest_seen_nlml_value is not None
            kernel_state_dict = best_seen_kernel_state_dict
        else:
            kernel_state_dict = kernel.state_dict()

        return kernel_state_dict

    def _fit(
        self,
        iters: int = 20,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        optimize_likelihood: bool = True,
        max_likelihood: float = 0.5,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.1}

        config_strings = self._get_config_strings(configs=self.x_configs)

        # Optimize each kernel and load its best seen state
        for kernel in self.kernels:
            kernel.train()
            best_seen_kernel_state_dict = self._optimize_kernel(
                kernel=kernel,
                configs=config_strings,
                iters=iters,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                likelihood=self.likelihood,
            )
            kernel.load_state_dict(state_dict=best_seen_kernel_state_dict)

        # Compute the individual best seen kernel results
        kernel_results = []
        for kernel in self.kernels:
            kernel.eval()
            K_i = kernel.forward(configs=config_strings)
            kernel_results.append(K_i)

        # Stack all the results into one tensor
        kernel_results = torch.stack(kernel_results, dim=0)
        assert kernel_results.shape == (
            len(self.kernels), len(self.x_configs), len(self.x_configs)
        )

        # Whether to include the likelihood (jitter or noise variance) as a hyperparameter
        likelihood = torch.tensor(self.likelihood)
        likelihood.requires_grad_(optimize_likelihood)

        # Whether likelihood and the parameters of the combining kernel are optimized
        optim_vars = []
        for var in [likelihood, *self.combining_kernel.parameters()]:
            if var is not None and var.is_leaf and var.requires_grad:
                optim_vars.append(var)
        _logger.debug("Vars to optimize: \n%r", optim_vars)

        nlml = None
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
            best_seen_combine_kernel_state_dict = None
            best_seen_likelihood_value = None
            lowest_seen_nlml_value = None

            # Prepare the results from the other kernels and
            #  set the combining kernel to train mode
            kernel_results = kernel_results.clone().detach()
            self.combining_kernel.train()

            # The 0th iteration uses the initial weights,
            #  the 1st iteration uses the weights after one optimization iteration,
            #  hence the `(iters + 1)` for `iters` optimization iterations
            for i in range(iters + 1):
                _logger.debug("Starting optimization iteration %d", i)

                with torch.no_grad():
                    # Clamp the likelihood to an appropriate range
                    _logger.debug("Clamping likelihood: before=%r", likelihood)
                    likelihood.clamp_(1e-5, max_likelihood)
                    _logger.debug("Clamping likelihood: after=%r", likelihood)

                # Reset the gradients
                optim.zero_grad(set_to_none=True)

                K = self.combining_kernel.forward(kernel_results=kernel_results)
                K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
                    K, likelihood, gpytorch_kinv=self.gpytorch_kinv
                )

                # Compute the loss
                nlml = -graph_gp_hierarchy.compute_log_marginal_likelihood(
                    K_i=K_i,
                    logDetK=logDetK,
                    y=self.y,
                )

                # Check if best seen results achieved and if so, store the values
                # Ensure no operations affecting gradients are done here
                iter_nlml = nlml.item()
                if lowest_seen_nlml_value is None or iter_nlml < lowest_seen_nlml_value:
                    best_seen_combine_kernel_state_dict = self.combining_kernel.state_dict()
                    best_seen_likelihood_value = likelihood.clone().detach()
                    lowest_seen_nlml_value = nlml.clone().detach()
                    _logger.debug("Optimization values improved in iteration %d", i)

                if i % 10 == 0:
                    _logger.debug(
                        "Iteration: %d/%d. (state_dict, likelihood, NLML): \n%r",
                        i, iters, (self.combining_kernel.state_dict(), likelihood, nlml)
                    )

                # Do a backward pass on the loss and then an optimization step
                nlml.backward()

                for var in optim_vars:
                    assert var.is_leaf, var
                    assert var.grad is not None, var
                    _logger.debug("Optim var and gradient: %s :: %s", var, var.grad)
                optim.step()

            # Now that the optimization is finished,
            #  set the variables to the best seen values for them
            assert best_seen_combine_kernel_state_dict is not None
            assert best_seen_likelihood_value is not None
            assert lowest_seen_nlml_value is not None
            combine_kernel_state_dict = best_seen_combine_kernel_state_dict
            likelihood = best_seen_likelihood_value
            nlml = lowest_seen_nlml_value

            # Set the combine state to the best found one
            self.combining_kernel.load_state_dict(combine_kernel_state_dict)

        # Do a forward pass
        # If optimization iterations were done, then the best seen values are used.
        # Otherwise, the initial model values are used
        kernel_results = kernel_results.clone().detach()
        self.combining_kernel.eval()
        K = self.combining_kernel.forward(kernel_results=kernel_results)
        K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
            K, likelihood, gpytorch_kinv=self.gpytorch_kinv
        )

        # Save the results of the fitting process
        self.likelihood = likelihood.item()
        self.nlml = nlml.clone().detach() if nlml is not None else None
        self.K = K.clone().detach()
        self.K_i = K_i.clone().detach()
        self.logDetK = logDetK.clone().detach()

        # Verify that the likelihood did not change if it should not have
        if not optimize_likelihood:
            assert self.likelihood == self.init_likelihood, (
                self.likelihood, self.init_likelihood
            )

        # Log the results
        _logger.debug("Optimization summary: ")
        _logger.debug("Best seen NLML: %r", self.nlml)
        _logger.debug("Combine weights: \n%r", list(self.combining_kernel.parameters()))
        _logger.debug(
            "Kernel weights: \n%r", [list(k.parameters()) for k in self.kernels]
        )
        _logger.debug("Likelihood: %r", self.likelihood)
        _logger.debug("%r computed K=\n%r", self.__class__.__name__, K)

    def predict(self, x_configs: Iterable):
        if self.K_i is None or self.logDetK is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. "
                + "Please first call the fit method to fit on the training data!"
            )

        x_configs = tuple(x_configs)
        if not x_configs:
            raise ValueError(f"No configs given to predict: {x_configs}")

        _logger.debug("%r is running predict on %d configs", self, len(x_configs))

        # All configs, including those from training are needed
        X_configs_all = self.x_configs + x_configs
        n_train_x_configs = len(self.x_configs)

        config_strings = self._get_config_strings(configs=X_configs_all)

        # No need to collect gradients while predicting
        with torch.no_grad():
            # Compute the individual best seen kernel results
            kernel_results = []
            for kernel in self.kernels:
                kernel.eval()
                K_i = kernel.forward(configs=config_strings)
                kernel_results.append(K_i)

            # Stack all the results into one tensor
            kernel_results = torch.stack(kernel_results, dim=0)
            assert kernel_results.shape == (
                len(self.kernels), len(X_configs_all), len(X_configs_all)
            )

            self.combining_kernel.eval()
            K_full = self.combining_kernel.forward(kernel_results=kernel_results)

        K_s = K_full[:n_train_x_configs:, n_train_x_configs:]
        K_ss = (
           K_full[n_train_x_configs:, n_train_x_configs:] +
           self.likelihood * torch.eye(len(x_configs))
        )

        mu_s = K_s.t() @ self.K_i @ self.y
        mu_s = graph_gp_hierarchy.unnormalize_y(mu_s, self.y_mean, self.y_std)

        cov_s = K_ss - K_s.t() @ self.K_i @ K_s
        cov_s = torch.clamp(cov_s, self.likelihood, np.inf)

        std_s = torch.sqrt(cov_s)
        std_s = graph_gp_hierarchy.unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s ** 2

        _logger.debug("%r finished predict on %d configs", self, len(x_configs))
        return mu_s, cov_s
