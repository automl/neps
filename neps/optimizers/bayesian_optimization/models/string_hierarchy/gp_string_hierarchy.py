from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from ...kernels.string_hierarchy import CombiningKernel
from .. import gp_hierarchy as graph_gp_hierarchy  # Needed for some helper functions

_logger = logging.getLogger(__name__)


def _softmax(values):
    return torch.exp(values) / torch.sum(torch.exp(values))


class GPStringHierarchy:
    def __init__(
        self,
        graph_kernels: Iterable,
        hp_kernels: Iterable | None = None,
        likelihood: float = 1e-3,
        combining_kernel_variant: str = "sum",
        normalize_combining_kernel: bool = True,
        surrogate_model_fit_args: dict = None,
        gpytorch_kinv: bool = False,
        vectorial_features: int | None = None,
    ):
        if hp_kernels:
            raise ValueError("HP kernels are not supported!")
        if vectorial_features:
            raise ValueError("Vectorial features are not supported!")

        self.kernels = tuple(graph_kernels)
        self.normalize_combining_kernel = normalize_combining_kernel
        self.combining_kernel_variant = combining_kernel_variant
        self.surrogate_model_fit_args = surrogate_model_fit_args or {}
        self.gpytorch_kinv = gpytorch_kinv

        # Set the initial likelihood
        self.init_likelihood = float(likelihood)

        # Initialise the weights to uniform
        self.init_weights = torch.tensor(
            [1.0 / len(self.kernels) for _ in self.kernels],
        )
        assert self.init_weights.shape == (len(self.kernels),)

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

        self.weights: torch.Tensor = self.init_weights.clone()
        self.likelihood: float | None = self.init_likelihood
        self.nlml: float | None = None

        self.K, self.K_i, self.logDetK = None, None, None

        self.combining_kernel = CombiningKernel(
            combining_kernel_variant=self.combining_kernel_variant,
            kernels=self.kernels,
        )

    @property
    def x(self):  # used upstream
        return self.x_configs

    def fit(self, train_x: Iterable, train_y: Iterable):
        train_x = tuple(train_x)
        train_y = tuple(train_y)
        self._reset_model_state(train_x=train_x, train_y=train_y)
        self._fit(**self.surrogate_model_fit_args)

    def _fit(
        self,
        iters: int = 20,
        optimizer: str = "adam",
        optimize_lik: bool = True,
        max_lik: float = 0.5,
        optimizer_kwargs: dict | None = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.1}

        weights = self.weights.clone().detach()
        likelihood = torch.tensor(self.likelihood)
        nlml = None

        # Whether weights will be optimized
        optimize_weights = len(self.kernels) > 1
        weights.requires_grad_(optimize_weights)

        # Whether to include the likelihood (jitter or noise variance) as a hyperparameter
        likelihood.requires_grad_(optimize_lik)

        optim_vars = []
        for var in [weights, likelihood]:
            if var is not None and var.is_leaf and var.requires_grad:
                optim_vars.append(var)
        _logger.debug("Vars to optimize: \n%r", optim_vars)

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
            lowest_seen_nlml_value = None
            best_seen_optim_var_values = None

            # The 0th iteration uses the initial weights,
            #  the 1st iteration uses the weights after one optimization iteration,
            #  hence the `(iters + 1)` for `iters` optimization iterations
            for i in range(iters + 1):
                _logger.debug("Starting optimization iteration %d", i)

                with torch.no_grad():
                    # Clamp the likelihood to an appropriate range
                    if likelihood is not None and likelihood.is_leaf:
                        _logger.debug("Clamping likelihood: before=%r", likelihood)
                        likelihood.clamp_(1e-5, max_lik)
                        _logger.debug("Clamping likelihood: after=%r", likelihood)

                # Reset the gradients
                optim.zero_grad(set_to_none=True)

                # Values should be in interval [0, 1] and sum up to 1
                # If weights got out of bounds during optimization
                #  put the weights in the right range
                # Create new variable to avoid in-place operation on leaf
                weights_in_range = _softmax(weights.clone())

                # Forward pass
                K = self.combining_kernel.transform(
                    weights=weights_in_range,
                    configs=self.x_configs,
                    normalize=self.normalize_combining_kernel,
                )
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
                    best_seen_optim_var_values = [
                        # For weights store their values in the right range
                        #  just as they were used in the combining kernel
                        # Otherwise we need to do the normalization
                        #  also during predicting
                        weights_in_range.clone().detach(),
                        likelihood.clone().detach(),
                    ]
                    _logger.debug("Optimization values improved in iteration %d", i)

                if i % 10 == 0:
                    _logger.debug(
                        "Iteration: %d/%d. (weights, likelihood, NLML): \n%r",
                        i, iters, (weights_in_range, likelihood, nlml)
                    )

                # Do a backward pass on the loss and then an optimization step
                nlml.backward()
                for var in optim_vars:
                    assert var.is_leaf, var
                    assert var.grad is not None, var
                optim.step()

            # Now that the optimization is finished,
            #  set the variables to the best seen values for them
            assert lowest_seen_nlml_value is not None
            assert best_seen_optim_var_values is not None
            nlml = lowest_seen_nlml_value
            weights, likelihood = best_seen_optim_var_values

        # Do a forward pass
        # If optimization iterations were done, then the best seen values are used.
        # Otherwise, the initial model values are used
        K = self.combining_kernel.transform(
            weights=weights,
            configs=self.x_configs,
            normalize=self.normalize_combining_kernel,
        )
        K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
            K, likelihood, gpytorch_kinv=self.gpytorch_kinv
        )

        # Save the results of the fitting process
        self.weights = weights
        self.likelihood = likelihood.item()
        self.nlml = nlml
        self.K = K.clone()
        self.K_i = K_i.clone()
        self.logDetK = logDetK.clone()

        # Verify that the likelihood did not change if it should not have
        if not optimize_lik:
            assert self.likelihood == self.init_likelihood, (
                self.likelihood, self.init_likelihood
            )

        # Log the results
        _logger.debug("Optimization summary: ")
        _logger.debug("Best seen NLML: %r", self.nlml)
        _logger.debug("Weights: \n%r", self.weights)
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

        # All configs, including those from training are needed
        X_configs_all = self.x_configs + x_configs
        n_train_x_configs = len(self.x_configs)
        _logger.debug("%r is running predict on %d configs", len(x_configs))

        K_full = self.combining_kernel.transform(
            weights=self.weights,
            configs=X_configs_all,
            normalize=self.normalize_combining_kernel,
        )

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
        cov_s = std_s**2

        _logger.debug("%r finished predict on %d configs", len(x_configs))
        return mu_s, cov_s
