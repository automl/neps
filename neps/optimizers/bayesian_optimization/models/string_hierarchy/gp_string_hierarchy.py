from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
import torch

from .basic_string_kernels import ASK
from .combining_string_kernels import CombiningKernel
from .. import gp_hierarchy as graph_gp_hierarchy  # Needed for some math helper functions

_logger = logging.getLogger(__name__)


class GPStringHierarchy:
    def __init__(
        self,
        graph_kernels: Iterable[ASK],
        hp_kernels: Iterable | None = None,
        likelihood: float = 1e-3,
        combining_kernel_variant: str = "sum",
        surrogate_model_fit_args: dict | None = None,
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

        # Reset the state
        self._reset_state()

    def _reset_state(
        self,
        train_x: tuple | None = None,
        train_y: tuple | None = None,
    ):
        # Filter out the configs for which the y value is infinite
        # Otherwise, we will get errors during math operations
        if train_y is not None:
            filtered_x = []
            filtered_y = []
            assert len(train_y) == len(train_x), (len(train_y), len(train_x))
            for x, y in zip(train_x, train_y):
                if not math.isinf(y):
                    filtered_x.append(x)
                    filtered_y.append(y)
                else:
                    _logger.debug("Skipped config with inf value: %s", x)
            train_x = tuple(filtered_x)
            train_y = tuple(filtered_y)
            assert len(train_y) == len(train_x), (len(train_y), len(train_x))

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

    def fit(self, train_x: Iterable, train_y: Iterable):
        train_x = tuple(train_x)
        train_y = tuple(train_y)
        self._reset_state(train_x=train_x, train_y=train_y)
        self._fit(**self.surrogate_model_fit_args)

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

        likelihood = self.likelihood

        # Optimize each kernel
        #  and compute the individual best seen kernel results
        kernel_results = []
        for kernel in self.kernels:
            K_i = kernel.fit_transform(
                train_x=self.x_configs,
                train_y=self.y,
                iters=iters,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                likelihood=likelihood,
                gpytorch_kinv=self.gpytorch_kinv,
            )
            kernel_results.append(K_i)

        # Stack all the results into one tensor
        kernel_results = torch.stack(kernel_results, dim=0)
        assert kernel_results.shape == (
            len(self.kernels), len(self.x_configs), len(self.x_configs)
        )

        # Optimize the combining kernel and compute
        #  the best seen result, likelihood and nlml
        (
            K,
            optimized_likelihood,
            optimized_nlml,
        ) = self.combining_kernel.fit_transform(
            train_x=kernel_results,
            train_y=self.y,
            iters=iters,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            optimize_likelihood=optimize_likelihood,
            max_likelihood=max_likelihood,
            likelihood=likelihood,
            gpytorch_kinv=self.gpytorch_kinv,
        )

        # Save the results of the fitting process
        self.K = K.clone().detach()
        self.likelihood = optimized_likelihood
        self.nlml = optimized_nlml
        K_i, logDetK = graph_gp_hierarchy.compute_pd_inverse(
            self.K, self.likelihood, gpytorch_kinv=self.gpytorch_kinv
        )
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

        # No need to collect gradients while predicting
        with torch.no_grad():
            # Compute the individual kernel results
            kernel_results = []
            for kernel in self.kernels:
                K_i = kernel.transform(x_configs=X_configs_all)
                kernel_results.append(K_i)

            # Stack all the results into one tensor
            kernel_results = torch.stack(kernel_results, dim=0)
            assert kernel_results.shape == (
                len(self.kernels), len(X_configs_all), len(X_configs_all)
            )

            # Combine the kernel results
            K_full = self.combining_kernel.transform(kernel_results=kernel_results)

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
