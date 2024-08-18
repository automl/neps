from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence
from typing_extensions import Literal
import torch.nn as nn

import numpy as np
import torch

from neps.optimizers.bayesian_optimization.kernels.kernel import (
    Kernel,
    log_marginal_likelihood,
    compute_pd_inverse,
)
from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import (
    HammingKernel,
    Matern52Kernel,
)
from neps.optimizers.bayesian_optimization.kernels.weisfilerlehman import (
    WeisfilerLehman,
)
from neps.search_spaces import SearchSpace
from neps.search_spaces.encoding import (
    IntegerCategoricalTransformer,
    JointTransformer,
    MinMaxNormalizer,
    OneHotEncoder,
    TensorTransformer,
    Transformer,
    WLInputTransformer,
)
from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.integer import IntegerParameter

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger(__name__)


# The optimization we do for the noise is relatively cheap while the matrices
NOISE_VARIANCE_GRID = (1e-6, 1e-4, 1e-2, 1, 1e1, 1e2)


@dataclass
class ComprehensiveGP:
    space: SearchSpace
    kernels: dict[str, tuple[Kernel, Transformer]]
    combined_kernel: Literal["sum", "product"] = "sum"

    noise_variance: Sequence[float] = NOISE_VARIANCE_GRID
    kernel_parameter_grid: Mapping[str, Sequence[Mapping[str, Any]]] | bool = True

    optimizer: Literal["adam", "sgd"] = "adam"
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"lr": 0.1})
    optimizer_iters: int = 20
    device: torch.device | None = None

    # Post fit attributes
    K_inv_: torch.Tensor | None = None
    n_train_: int | None = None
    likelihood_: float | None = None
    y_: torch.Tensor | None = None
    y_normalized_: torch.Tensor | None = None
    y_mean_: float | None = None
    y_std_: float | None = None
    optimized_kernels_: dict[str, Kernel] | None = None
    train_data_: dict[str, Any] | None = None

    def __post_init__(self):
        # TODO: Remove when search space is just definition and does not hold values.
        self.space = self.space.clone()

    def fit(self, x: list[dict[str, Any]], train_y: torch.Tensor) -> None:
        # Preprocessing
        y_ = torch.as_tensor(train_y, device=self.device, dtype=torch.float64)

        # TODO: Dunno if I like this silent hack, setting std to 1 if no std
        self.y_std_ = s if (s := torch.std(y_).item()) != 0 else 1
        self.y_mean_ = torch.mean(y_).item()
        self.y_normalized_ = (y_ - self.y_mean_) / self.y_std_
        self.y_ = y_

        _data = {
            key: transformer.encode(x, self.space)
            for key, (_, transformer) in self.kernels.items()
        }

        # optimized kernel parameters + noise variance
        optim_vars: list[nn.Parameter] = []

        grids = {
            name: k.suggested_grid
            for name, (k, _) in self.kernels.items()
            if k.suggested_grid is not None
        }

        kernels: dict[str, Kernel] = {}
        for kernel_name, (kernel, _) in self.kernels.items():
            xs = _data[kernel_name]
            grid = grids[kernel_name]

            maybe_optimized_kernel = kernel.grid_search(
                x=xs,
                y=self.y_normalized_,
                grid=grid,
            )
            if isinstance(maybe_optimized_kernel, Exception):
                raise ValueError(
                    f"Failed to optimize kernel {kernel_name} with grid {grid}."
                ) from maybe_optimized_kernel

            opt_kernel, _ = maybe_optimized_kernel
            gradient_enabled_kernel = opt_kernel.as_optimizable()
            kernels[kernel_name] = gradient_enabled_kernel

            optim_vars.extend(gradient_enabled_kernel.parameters())

        # Now that we've optimized the kernels, we convert go convert their
        # parameters into a tensor we can further refine with some optimizer iterations
        # - Optimize kernel-lengthscales, kernel-outputscale, noise-variance
        #   and any additional parameters they wish to advertise.
        noise_variance = nn.Parameter(
            torch.tensor(1e-3, device=self.device, dtype=torch.float64)
        )
        optim_vars.append(noise_variance)

        if self.optimizer == "adam":
            optim = torch.optim.Adam(optim_vars, **self.optimizer_kwargs)  # type: ignore
        elif self.optimizer == "sgd":
            optim = torch.optim.SGD(optim_vars, **self.optimizer_kwargs)  # type: ignore
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

        K_inv: torch.Tensor | None = None
        N = len(x)
        for i in range(self.optimizer_iters):
            optim.zero_grad()
            # Now we iterate over kernels to build up K
            _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
            K = _init(N, N, device=self.device, dtype=torch.float64)
            for kernel_name, kernel in kernels.items():
                data = _data[kernel_name]
                gram = kernel.forward(data, data)

                if self.combined_kernel == "sum":
                    K.add_(gram)
                else:
                    K.mul_(gram)

            K.diag().add_(noise_variance)

            K_inv, logDetK = compute_pd_inverse(K)
            nlml = -log_marginal_likelihood(K_inv, logDetK, y=self.y_normalized_)

            # TODO: Could early stop here...
            nlml.backward()
            optim.step()

            with torch.no_grad():
                noise_variance.clamp_(1e-6, np.inf)

        # Apply the optimal hyperparameters
        assert K_inv is not None
        self.K_inv_ = K_inv.clone()
        self.noise_variance_ = noise_variance.item()
        self.optimized_kernels_ = kernels
        self.n_train_ = N
        self.train_data_ = _data

    def predict(self, x: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Kriging predictions."""
        if (
            self.K_inv_ is None
            or self.n_train_ is None
            or self.optimized_kernels_ is None
            or self.train_data_ is None
            or self.y_normalized_ is None
            or self.y_std_ is None
        ):
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )
        _data = {
            key: transformer.encode(x, self.space)
            for key, (_, transformer) in self.kernels.items()
        }

        _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
        n_test = len(x)

        K_train_test = _init(
            self.n_train_, n_test, device=self.device, dtype=torch.float64
        )
        K_test_test = _init(n_test, n_test, device=self.device, dtype=torch.float64)

        for kernel_name, kernel in self.optimized_kernels_.items():
            train_x = self.train_data_[kernel_name]
            test_x = _data[kernel_name]

            gram = kernel.forward(train_x, test_x)
            if self.combined_kernel == "sum":
                K_train_test.add_(gram)
            else:
                K_train_test.mul_(gram)

            gram = kernel.forward(test_x, test_x)
            if self.combined_kernel == "sum":
                K_test_test.add_(gram)
            else:
                K_test_test.mul_(gram)

        # Compute the predictive mean

        # Scale by the standard deviation and mean
        mu_s = K_train_test.t() @ self.K_inv_ @ self.y_normalized_
        mu_s = mu_s * self.y_std_ + self.y_mean_

        cov_s = K_test_test - K_train_test.t() @ self.K_inv_ @ K_train_test
        cov_s.diagonal().clamp_(self.noise_variance_, np.inf)
        cov_s *= self.y_std_**2

        return mu_s, cov_s

    @classmethod
    def get_default(
        cls, space: SearchSpace, *, include_fidelities: bool = False
    ) -> ComprehensiveGP:
        kernels = get_default_kernels(space=space, include_fidelities=include_fidelities)
        return cls(space=space, kernels=kernels)


def get_default_kernels(
    *,
    space: SearchSpace,
    include_fidelities: bool = False,
) -> dict[str, tuple[Kernel, Transformer]]:
    kernels: dict[str, tuple[Kernel, Transformer]] = {}

    # We will always need to use a graph kernel for graphs and there's no
    # possibility to embed them into a tensor.
    if any(space.graphs):
        for hp_name in space.graphs:
            kernels[f"graph_{hp_name}"] = (
                WeisfilerLehman(h=2, oa=True),
                WLInputTransformer((hp_name,)),
            )

    assert all(
        isinstance(f, (IntegerParameter, FloatParameter)) for f in space.fidelities
    ), "Assumption for numeric represetnation of fidelity broken"

    any_numerical = any(space.numerical) or (include_fidelities and any(space.fidelities))
    if any_numerical:
        # At least one numerical, fuse numeric + categoricals into one tensor encoding
        transformers: list[TensorTransformer] = []
        if any(space.categoricals):
            transformers.append(OneHotEncoder(tuple(space.categoricals)))

        if include_fidelities:
            min_max_normalizer = MinMaxNormalizer(
                tuple(space.numerical) + tuple(space.fidelities)
            )
        else:
            min_max_normalizer = MinMaxNormalizer(tuple(space.numerical))

        transformers.append(min_max_normalizer)
        kernels["vectorial"] = (Matern52Kernel(), JointTransformer.join(*transformers))
    else:
        # At this point, we assume only categoricals and maybe fidelities
        assert any(space.categoricals)

        if include_fidelities and any(space.fidelities):
            fid_normalizer = MinMaxNormalizer(tuple(space.fidelities))
            one_hot_encoder = OneHotEncoder(tuple(space.categoricals))

            transformer = JointTransformer.join(one_hot_encoder, fid_normalizer)
            kernels["vectorial"] = (Matern52Kernel(), transformer)
        else:
            transformer = IntegerCategoricalTransformer(tuple(space.categoricals))
            kernels["categorical"] = (HammingKernel(), transformer)

    return kernels
