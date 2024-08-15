from __future__ import annotations

import logging
import torch
import numpy as np
from typing import Literal, Sequence, Any, Mapping
from typing_extensions import Literal
from dataclasses import dataclass, field
from itertools import product

from neps.optimizers.bayesian_optimization.kernels.kernel import (
    Kernel,
    NumericKernel,
    compute_normalized_log_marginal_likelihood,
    compute_pd_inverse,
)

from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import Stationary
from neps.optimizers.bayesian_optimization.kernels.weisfilerlehman import (
    WeisfilerLehman,
)
from neps.search_spaces.encoding import TensorEncodedConfigs
from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger(__name__)

f64 = torch.float64


GRID_WL_LENGTHSCALES = torch.tensor([np.e**i for i in range(-2, 3)], dtype=f64)
GRID_WL_SUBTREE_CANDIDATES = (1, 2, 3, 4, 5)


def _default_param_grid() -> dict[type[Kernel], list[dict[str, Any]]]:
    return {
        WeisfilerLehman: [
            {"h": h, "se_kernel": Stationary(lengthscale=l)}
            for h, l in product(GRID_WL_SUBTREE_CANDIDATES, GRID_WL_LENGTHSCALES)
        ]
    }


@dataclass
class ComprehensiveGP:
    space: SearchSpace
    kernels: Sequence[tuple[Kernel, Sequence[str]]]
    combined_kernel: Literal["sum", "product"] = "sum"
    initial_likelihood: float = 1e-3
    optimize_likelihood: bool = True
    max_likelihood: float = 0.01
    optimizer: Literal["adam", "sgd"] = "adam"
    optimizer_iters: int = 20
    optimize_wl_layer_weights: bool = False
    surrogate_model_fit_args: Mapping[str, Any] = field(default_factory=dict)
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"lr": 0.1})
    kernel_hp_grids: Mapping[type[Kernel], Sequence[Mapping[str, Any]]] = field(
        default_factory=_default_param_grid
    )

    # Post fit attributes
    K_i_: torch.Tensor | None = None
    n_train_: int | None = None
    likelihood_: float | None = None
    y_: torch.Tensor | None = None
    y_normalized_: torch.Tensor | None = None
    y_mean_: float | None = None
    y_std_: float | None = None
    optimized_kernels_: (
        list[tuple[NumericKernel | WeisfilerLehman, Sequence[str]]] | None
    ) = None
    kernel_weights_: torch.Tensor | None = None

    def __post_init__(self):
        # TODO: Remove when search space is just definition and does not hold values.
        self.space = self.space.clone()

    def fit(self, x: TensorEncodedConfigs, train_y: torch.Tensor) -> None:
        # Preprocessing
        y_ = torch.as_tensor(train_y, dtype=f64)

        # TODO: Dunno if I like this silent hack, setting std to 1 if no std
        self.y_std_ = s if (s := torch.std(y_).item()) != 0 else 1
        self.y_mean_ = torch.mean(y_).item()
        self.y_normalized_ = (y_ - self.y_mean_) / self.y_std_

        optimized_kernels: list[
            tuple[NumericKernel | WeisfilerLehman, Sequence[str]]
        ] = []
        _grids = self.kernel_hp_grids

        def _eval_kernel(_K: torch.Tensor) -> float:
            assert y_ is not None
            K_i, logDetK = compute_pd_inverse(_K)
            nlml = -compute_normalized_log_marginal_likelihood(K_i, logDetK, y_)
            return float(nlml)

        for kernel, hps in self.kernels:
            if isinstance(kernel, WeisfilerLehman):
                assert len(hps) == 1, "Only support single kernel per graph."
                _xs = x.wl_graph_input(hps[0])
            elif isinstance(kernel, NumericKernel):
                _xs = x.tensor(hps)
            else:
                raise ValueError(f"Unsupported kernel type {type(kernel)}")

            grid = next((g for t, g in _grids.items() if isinstance(kernel, t)), None)
            if grid is None:
                optimized_kernel = kernel.clone()
                _ = optimized_kernel.fit_transform(_xs)  # type: ignore
                optimized_kernels.append((kernel, hps))
                continue

            optimized_kernel, _ = kernel.grid_search(
                x=_xs,  # type: ignore
                grid=grid,
                to_minimize=_eval_kernel,
            )
            optimized_kernels.append((optimized_kernel, hps))

        # Optimization weights
        likelihood = torch.tensor(
            self.initial_likelihood,
            requires_grad=self.optimize_likelihood,
        )

        kernel_weights = torch.ones(
            len(optimized_kernels),
            requires_grad=len(optimized_kernels) > 1,
            dtype=f64,
        )
        should_optimize = lambda p: p.is_leaf and p.requires_grad

        # Linking the optimizer variables to the sum kernel
        optim_vars: list[torch.Tensor] = [
            a
            for a in (kernel_weights, likelihood)
            if a is not None and should_optimize(a)
        ]
        layer_weights = [
            kernel.layer_weights_
            for kernel, _ in optimized_kernels
            if isinstance(kernel, WeisfilerLehman)
            and kernel.layer_weights_ is not None
            and should_optimize(kernel.layer_weights_)
        ]
        lengthscales = [
            kernel.layer_weights_
            for kernel, _ in optimized_kernels
            if isinstance(kernel, NumericKernel) and should_optimize(kernel.lengthscale)
        ]
        lengthscalebounds = [
            kernel.lengthscale_bounds
            for kernel, _ in optimized_kernels
            if isinstance(kernel, NumericKernel) and should_optimize(kernel.lengthscale)
        ]

        # Select the optimizer
        if self.optimizer == "adam":
            optim = torch.optim.Adam(optim_vars, **self.optimizer_kwargs)  # type: ignore
        elif self.optimizer == "sgd":
            optim = torch.optim.SGD(optim_vars, **self.optimizer_kwargs)  # type: ignore
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

        K: torch.Tensor | None = None
        N = len(x)
        for _ in range(self.optimizer_iters):
            optim.zero_grad()

            # Now we iterate over kernels to build up K
            _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
            K = _init(N, N, dtype=f64)
            for (kernel, hps), weight in zip(self.kernels, kernel_weights):
                if isinstance(kernel, WeisfilerLehman):
                    assert len(hps) == 1, "Only support single kernel per graph."
                    _xs = x.wl_graph_input(hps[0])
                    gram = kernel.fit_transform(_xs)
                elif isinstance(kernel, NumericKernel):
                    _xs = x.tensor(hps)
                    gram = kernel.fit_transform(_xs)
                else:
                    raise ValueError(f"Unsupported kernel type {type(kernel)}")

                if self.combined_kernel == "sum":
                    K.add_(weight * gram)
                elif self.combined_kernel == "product":
                    K.mul_(weight * gram)
                else:
                    raise ValueError(f"Invalid combined_kernel {self.combined_kernel}")

            # Normalize
            K_diag = torch.sqrt(torch.diag(K))
            K /= torch.ger(K_diag, K_diag)
            K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)

            # If there's nothing to optimize, break out early
            if len(optim_vars) == 0:
                break

            nlml = -compute_normalized_log_marginal_likelihood(
                K_i, logDetK, y=self.y_normalized_
            )
            nlml.backward()
            optim.step()

            with torch.no_grad():
                kernel_weights.clamp_(0.0, 1.0)
                if likelihood.is_leaf:
                    likelihood.clamp_(1e-9, self.max_likelihood)

                for ls, ls_bounds in zip(lengthscales, lengthscalebounds):
                    ls.clamp_(*ls_bounds)

                for lw in layer_weights:
                    lw.clamp_(0.0, 1.0)

            optim.zero_grad()

        assert K is not None
        K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)

        # Apply the optimal hyperparameters
        self.K_i_ = K_i.clone()
        self.likelihood_ = likelihood.item()
        self.optimized_kernels_ = optimized_kernels
        self.kernel_weights_ = kernel_weights
        self.n_train_ = N

    def predict(self, x: TensorEncodedConfigs) -> tuple[torch.Tensor, torch.Tensor]:
        """Kriging predictions"""
        if self.K_i_ is None or self.n_train_ is None or self.kernel_weights_ is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )

        _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
        N = self.n_train_ + len(x)
        K = _init(N, N, dtype=f64)
        for (kernel, hps), weight in zip(self.kernels, self.kernel_weights_):
            if isinstance(kernel, WeisfilerLehman):
                assert len(hps) == 1, "Only support single kernel per graph."
                _x_test = x.wl_graph_input(hps[0])
                gram = kernel.transform(_x_test)
            elif isinstance(kernel, NumericKernel):
                _x_test = x.tensor(hps)
                gram = kernel.fit_transform(_x_test)
            else:
                raise ValueError(f"Unsupported kernel type {type(kernel)}")

            if self.combined_kernel == "sum":
                K.add_(weight * gram)
            elif self.combined_kernel == "product":
                K.mul_(weight * gram)
            else:
                raise ValueError(f"Invalid combined_kernel {self.combined_kernel}")

        # Concatenate the full list
        n_test = len(x)

        K_s = K[: self.n_train_ :, self.n_train_ :]
        K_ss = K[self.n_train_ :, self.n_train_ :] + self.likelihood_ * torch.eye(n_test)

        mu_s = K_s.t() @ self.K_i_ @ self.y_normalized_
        mu_s = mu_s * self.y_std_ + self.y_mean_

        cov_s = K_ss - K_s.t() @ self.K_i_ @ K_s
        cov_s = torch.clamp(cov_s, self.likelihood_, np.inf)
        cov_s = (torch.sqrt(cov_s) * self.y_std_) ** 2

        return mu_s, cov_s
