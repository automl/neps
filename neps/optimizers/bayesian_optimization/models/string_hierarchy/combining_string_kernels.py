from __future__ import annotations

import copy
import logging
from typing import Iterator

import torch

from ...kernels.string_hierarchy import combining_string_kernels
from .. import gp_hierarchy as graph_gp_hierarchy  # Needed for math helper functions

_logger = logging.getLogger(__name__)


# CombiningKernel

class CombiningKernel:
    def __init__(
        self,
        n_kernels: int,
        combining_kernel_variant: str,
    ):
        if n_kernels < 1:
            raise ValueError(f"n_kernels can not be < 1: {n_kernels}")
        self.n_kernels = n_kernels

        if combining_kernel_variant not in ("sum", "product"):
            raise ValueError(
                "Invalid value for combining kernel variant: "
                + f"{combining_kernel_variant!r}"
            )
        self.combining_kernel_variant = combining_kernel_variant

        # Reset the model state
        self._reset_state()

    def _reset_state(self):
        self._kernel = combining_string_kernels.CombiningKernel(
            n_kernels=self.n_kernels,
            combining_kernel_variant=self.combining_kernel_variant,
        )

    def _optimize_kernel(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        iters: int,
        optimizer: str,
        optimizer_kwargs: dict | None,
        optimize_likelihood: bool,
        max_likelihood: float,
        likelihood: float,
        gpytorch_kinv: bool,
    ) -> tuple[dict, float, float | None]:
        # Whether to include the likelihood (jitter or noise variance) as a hyperparameter
        likelihood = torch.tensor(likelihood)
        likelihood.requires_grad_(optimize_likelihood)

        # Whether likelihood and the parameters of the kernel are optimized
        optim_vars = []
        for var in [likelihood, *self._kernel.parameters()]:
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
            best_nlml_value = None
            best_kernel_state_dict = None
            best_likelihood_value = None

            # The 0th iteration uses the initial weights,
            #  the 1st iteration uses the weights after one optimization iteration,
            #  hence the `(iters + 1)` for `iters` optimization iterations
            for i in range(iters + 1):
                _logger.debug("Starting optimization iteration %d", i)

                with torch.no_grad():
                    # Clamp the likelihood to an appropriate range
                    torch.clamp_(likelihood, min=1e-5, max=max_likelihood)

                # Reset the gradients
                optim.zero_grad(set_to_none=True)

                # Forward pass
                self._kernel.train()
                K = self._kernel.forward(kernel_results=train_x)
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
                    best_likelihood_value = likelihood.clone().detach()
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
            assert best_likelihood_value is not None
            assert best_nlml_value is not None
            kernel_state_dict = best_kernel_state_dict
            likelihood = best_likelihood_value.item()
            nlml = best_nlml_value.item()
        else:
            kernel_state_dict = self._kernel.state_dict()
            likelihood = likelihood
            nlml = None

        return kernel_state_dict, likelihood, nlml

    def fit_transform(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        iters: int,
        optimizer: str,
        optimizer_kwargs: dict | None,
        optimize_likelihood: bool,
        max_likelihood: float,
        likelihood: float,
        gpytorch_kinv: bool,
    ) -> tuple[torch.Tensor, float, float]:
        train_x = train_x.clone().detach()
        train_y = train_y.clone().detach()
        optimizer_kwargs = {"weight_decay": 0.1, **optimizer_kwargs}

        self._reset_state()

        (
            optimized_kernel_state_dict,
            optimized_likelihood,
            optimized_nlml,
        ) = self._optimize_kernel(
            train_x=train_x,
            train_y=train_y,
            iters=iters,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            optimize_likelihood=optimize_likelihood,
            max_likelihood=max_likelihood,
            likelihood=likelihood,
            gpytorch_kinv=gpytorch_kinv,
        )
        _logger.debug("Loading optimized kernel state: %s", optimized_kernel_state_dict)
        self._kernel.load_state_dict(state_dict=optimized_kernel_state_dict)

        K = self.transform(kernel_results=train_x)
        return K, optimized_likelihood, optimized_nlml

    def transform(
        self,
        kernel_results: torch.Tensor,
    ) -> torch.Tensor:
        self._kernel.eval()
        K = self._kernel.forward(kernel_results=kernel_results)
        return K

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return list(self._kernel.parameters())
