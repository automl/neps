from __future__ import annotations

import logging
from copy import deepcopy
from typing import Iterable, Literal, Sequence, Any

import numpy as np
import contextlib
import torch

from neps.optimizers.bayesian_optimization.kernels.combine_kernels import (
    ProductKernel,
    SumKernel,
)

from neps.optimizers.bayesian_optimization.kernels.graph_kernel import GraphKernels
from neps.optimizers.bayesian_optimization.kernels.utils import extract_configs
from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import Stationary
from neps.optimizers.bayesian_optimization.kernels.weisfilerlehman import WeisfilerLehman
from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger(__name__)


class ComprehensiveGP:
    def __init__(
        self,
        space: SearchSpace,
        graph_kernels: Iterable,
        hp_kernels: Iterable,
        initial_likelihood: float = 1e-3,
        weights: Sequence[float] | torch.Tensor | None = None,
        combined_kernel: Literal["sum", "product"] = "sum",
        surrogate_model_fit_args: dict | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        wl_subtree_candidates: Sequence[int] = (1, 2, 3, 4, 5),
        wl_lengthscales: Sequence[float] = tuple(np.e**i for i in range(-2, 3)),
        optimize_likelihood: bool = True,
        optimizer: Literal["adam", "sgd"] = "adam",
        optimizer_iters: int = 20,
        max_likelihood: float = 0.01,
        optimize_wl_layer_weights: bool = False,
    ):
        graph_kernels = list(graph_kernels)
        hp_kernels = list(hp_kernels)
        n_graph_kernels = len(graph_kernels)
        n_vector_kernels = len(hp_kernels)
        n_kernels = n_graph_kernels + n_vector_kernels
        domain_kernels = [*graph_kernels, *hp_kernels]

        fixed_weights = weights is not None
        if weights is not None:
            if weights is not None:
                assert len(weights) == n_kernels, (
                    "the weights vector, if supplied, needs to have the same length as "
                    "the number of kernel_operators!"
                )
            init_weights = torch.as_tensor(weights).flatten()
        else:
            uniform_weight = 1.0 / self.n_kernels
            init_weights = torch.full((n_kernels,), uniform_weight, dtype=torch.float64)

        if combined_kernel == "product":
            _combined_kernel = ProductKernel(*domain_kernels, weights=weights)
        elif combined_kernel == "sum":
            _combined_kernel = SumKernel(*domain_kernels, weights=weights)
        else:
            raise NotImplementedError(
                f'Combining kernel {combined_kernel} is not yet implemented! Only "sum" '
                f'or "product" are currently supported. '
            )

        # TODO: Clone only needed while it can act like configurations
        self.space = space.clone()
        self.init_weights = init_weights
        self.fixed_weights = fixed_weights
        self.combined_kernel = _combined_kernel
        self.initial_likelihood = initial_likelihood
        self.surrogate_model_fit_args = surrogate_model_fit_args or {}
        self.domain_kernels: list = [*graph_kernels, *hp_kernels]
        self.n_kernels: int = len(self.domain_kernels)
        self.n_graph_kernels: int = len(graph_kernels)
        self.n_vector_kernels: int = len(hp_kernels)
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.1}
        self.optimize_likelihood = optimize_likelihood
        self.optimize_wl_layer_weights = optimize_wl_layer_weights
        self.optimizer = optimizer
        self.optimizer_iters = optimizer_iters
        self.max_likelihood = max_likelihood
        self.wl_subtree_candidates = wl_subtree_candidates
        self.wl_lengthscales = wl_lengthscales

        # Cache the Gram matrix inverse and its log-determinant
        self.K_ = None
        self.K_i_ = None
        self.logDetK_ = None
        self.theta_vector_ = None
        self.layer_weights_ = None
        self.nlml_ = None
        self.likelihood_: float | None = None
        self.weights_: torch.Tensor | None = None
        self.x_configs_: list[SearchSpace] | None = None
        self.y_: torch.Tensor | None = None
        self.y_normalized_: torch.Tensor | None = None
        self.y_mean_: float | None = None
        self.y_std_: float | None = None
        self.n_: int | None = None

    def fit(self, train_x: list[SearchSpace], train_y: list[float]) -> None:
        """Called by self.fit"""
        self.x_configs = train_x
        self.n_ = len(train_x)
        self.y_ = torch.as_tensor(train_y, dtype=torch.float64)

        # TODO: Dunno if I like this silent hack, setting std to 1 if no std
        self.y_std_ = s if (s := torch.std(self.y_).item()) != 0 else 1
        self.y_mean_ = torch.mean(self.y_).item()
        self.y_normalized_ = (self.y_ - self.y_mean_) / self.y_std_

        # The Gram matrix of the training data
        self.K_i_, self.logDetK_ = None, None

        if len(self.wl_subtree_candidates) > 0:
            graphs, _ = extract_configs(self.x_configs)
            graph_kernels = [
                k for k in self.domain_kernels if isinstance(k, GraphKernels)
            ]
            for i, kernel in enumerate(graph_kernels):
                if not isinstance(kernel, WeisfilerLehman):
                    logger.warning(f"No kernel opt. for {type(kernel).__name__}.")
                    continue

                _xs = (
                    [x[i] for x in graphs]
                    if isinstance(graphs[0], list)
                    else [x for x in graphs]
                )
                _grid_search_wl_kernel(
                    kernel=kernel,
                    subtree_candidates=self.wl_subtree_candidates,
                    train_x=_xs,
                    train_y=self.y_,
                    likelihood=self.initial_likelihood,
                    lengthscales=self.wl_lengthscales,
                )

        weights = self.init_weights.clone()

        if not self.fixed_weights and self.n_kernels > 1:
            weights.requires_grad_(True)

        n_cat = len(self.space.categoricals)
        n_num = len(self.space.numerical)
        theta_categorical = torch.ones(
            n_cat, requires_grad=n_cat > 1, dtype=torch.float64
        )
        theta_numerical = torch.ones(n_num, requires_grad=n_num > 1, dtype=torch.float64)

        theta_vectors = {
            "categorical": theta_categorical,
            "continuous": theta_numerical,  # NOTE: This actually includes integers too -_-
        }
        likelihood = torch.tensor(
            self.initial_likelihood, requires_grad=self.optimize_likelihood
        )

        layer_weights = None
        if self.optimize_wl_layer_weights:
            for kernel in self.domain_kernels:
                if isinstance(kernel, WeisfilerLehman) and kernel.h != 0:
                    layer_weights = torch.ones(kernel.h + 1, requires_grad=True)
                    break

        # Linking the optimizer variables to the sum kernel
        optim_vars = [
            a
            for a in (
                weights,
                likelihood,
                layer_weights,
                theta_categorical,
                theta_numerical,
            )
            if a is not None and a.is_leaf and a.requires_grad
        ]

        nlml = None
        if len(optim_vars) == 0:  # Skip optimisation
            K = self.combined_kernel.fit_transform(
                weights,
                self.x_configs,
                feature_lengthscale=theta_vectors,
                layer_weights=layer_weights,
                rebuild_model=True,
            )
            K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)
        else:
            # Select the optimizer
            if self.optimizer == "adam":
                optim = torch.optim.Adam(optim_vars, **self.optimizer_kwargs)  # type: ignore
            elif self.optimizer == "sgd":
                optim = torch.optim.SGD(optim_vars, **self.optimizer_kwargs)  # type: ignore
            else:
                raise ValueError(f"Invalid optimizer {self.optimizer}")

            K: torch.Tensor | None = None
            for i in range(self.optimizer_iters):
                optim.zero_grad()
                K = self.combined_kernel.fit_transform(
                    weights=weights,
                    configs=train_x,  # TODO
                    feature_lengthscale=theta_vectors,
                    layer_weights=layer_weights,
                    rebuild_model=True,
                    save_gram_matrix=True,
                )
                K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)
                nlml = -compute_log_marginal_likelihood(
                    K_i, logDetK, y=self.y_normalized_
                )
                nlml.backward()
                if i % 10 == 0:
                    logger.debug(
                        f"Iteration: {i}/{self.optimizer_iters} "
                        f"Negative log-marginal likelihood:"
                        f"{nlml.item()} {theta_vectors} {weights} {likelihood}"
                    )

                optim.step()  # TODO

                with torch.no_grad():
                    if weights.is_leaf:
                        weights.clamp_(0.0, 1.0)

                    theta_vectors = self.combined_kernel.clamp_theta_vector(theta_vectors)

                    if likelihood.is_leaf:
                        likelihood.clamp_(1e-5, self.max_likelihood)

                    if layer_weights is not None and layer_weights.is_leaf:
                        layer_weights.clamp_(0.0, 1.0)

                optim.zero_grad(set_to_none=True)

            assert K is not None
            K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)

        # Apply the optimal hyperparameters
        self.weights_ = weights.clone() / torch.sum(weights)
        self.K_i_ = K_i.clone()
        self.K_ = K.clone()
        self.logDetK_ = logDetK.clone()
        self.likelihood_ = likelihood.item()
        self.theta_vector_ = theta_vectors
        self.layer_weights_ = layer_weights
        self.nlml_ = nlml.detach().cpu() if nlml is not None else None

        for kernel in self.combined_kernel.kernels:
            if isinstance(kernel, Stationary):
                kernel.update_hyperparameters(lengthscale=self.theta_vector_)

        logger.debug("Optimisation summary: ")
        logger.debug(f"Optimal NLML: {nlml}")
        logger.debug(f"Lengthscales: {theta_vectors}")
        with contextlib.suppress(AttributeError):
            logger.debug(f"Optimal h: {self.domain_kernels[0]._h}")
        logger.debug(f"Weights: {self.weights_}")
        logger.debug(f"Lik: {self.likelihood_}")
        logger.debug(f"Optimal layer weights {layer_weights}")

    def predict(self, x_configs: list[SearchSpace]) -> tuple[torch.Tensor, torch.Tensor]:
        """Kriging predictions"""
        if not isinstance(x_configs, list):
            x_configs = [x_configs]

        if self.K_i_ is None or self.logDetK_ is None or self.weights_ is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )

        # Concatenate the full list
        X_configs_all = self.x_configs + x_configs
        n_train = len(self.x_configs)
        n_test = len(x_configs)

        K_full = self.combined_kernel.fit_transform(
            weights=self.weights_,
            configs=X_configs_all,
            layer_weights=self.layer_weights_,
            feature_lengthscale=self.theta_vector_,
            rebuild_model=True,
            save_gram_matrix=False,
            gp_fit=False,
        )

        K_s = K_full[:n_train:, n_train:]
        K_ss = K_full[n_train:, n_train:] + self.likelihood_ * torch.eye(n_test)

        mu_s = K_s.t() @ self.K_i_ @ self.y_
        mu_s = mu_s * self.y_std_ + self.y_mean_

        cov_s = K_ss - K_s.t() @ self.K_i_ @ K_s
        cov_s = torch.clamp(cov_s, self.likelihood_, np.inf)
        cov_s = (torch.sqrt(cov_s) * self.y_std_) ** 2

        return mu_s, cov_s


def _grid_search_wl_kernel(
    kernel: WeisfilerLehman,
    subtree_candidates,
    train_x: list,
    train_y: torch.Tensor,
    likelihood: float,
    lengthscales=None,
):
    """Optimize the *discrete hyperparameters* of Weisfeiler Lehman kernel.
    k: a Weisfeiler-Lehman kernel instance
    hyperparameter_candidate: list of candidate hyperparameter to try
    train_x: the train data
    train_y: the train label
    lik: likelihood
    lengthscale: if using RBF kernel for successive embedding, the list of lengthscale to be grid searched over
    """
    # lik = 1e-6
    assert len(train_x) == len(train_y)
    best_nlml = torch.tensor(np.inf)
    best_subtree_depth = None
    best_lengthscale = None
    best_K = None
    if lengthscales is not None and kernel.se is not None:
        candidates = [(h_, l_) for h_ in subtree_candidates for l_ in lengthscales]
    else:
        candidates = [(h_, None) for h_ in subtree_candidates]

    for i in candidates:
        if kernel.se is not None:
            kernel.change_se_params({"lengthscale": i[1]})

        kernel.change_kernel_params({"h": i[0]})
        K = kernel.fit_transform(train_x, rebuild_model=True, save_gram_matrix=True)
        K_i, logDetK = compute_pd_inverse(K, jitter=likelihood)
        nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
        if nlml < best_nlml:
            best_nlml = nlml
            best_subtree_depth, best_lengthscale = i
            best_K = torch.clone(K)

    kernel.change_kernel_params({"h": best_subtree_depth})
    if kernel.se is not None:
        kernel.change_se_params({"lengthscale": best_lengthscale})
    kernel._gram = best_K


def compute_log_marginal_likelihood(
    K_i: torch.Tensor,
    logDetK: torch.Tensor,
    y: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the zero mean Gaussian process log marginal likelihood given the inverse of Gram matrix K(x2,x2), its
    log determinant, and the training label vector y.
    Option:

    normalize: normalize the log marginal likelihood by the length of the label vector, as per the gpytorch
    routine.
    """
    lml = (
        -0.5 * (y.t() @ K_i @ y)
        + 0.5 * logDetK
        - y.shape[0] / 2.0 * torch.log(2 * torch.tensor(np.pi))
    )
    return lml / y.shape[0] if normalize else lml


def compute_pd_inverse(
    K: torch.Tensor,
    *,
    jitter: float | torch.Tensor = 1e-6,
    attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion."""
    n = K.shape[0]
    assert (
        isinstance(jitter, float) or jitter.ndim == 0
    ), "only homoscedastic noise variance is allowed here!"
    for i in range(attempts):
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10**i
            Kc = torch.linalg.cholesky(K + jitter_diag)
            break
        except RuntimeError:
            pass
    else:
        raise RuntimeError(f"Gram matrix not positive definite despite of jitter:\n{K}")

    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.to(dtype=torch.float64), logDetK.to(dtype=torch.float64)
