from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam  # type: ignore

from neps.optimizers.bayesian_optimization.kernels.kernel import (
    Kernel,
    compute_pd_inverse,
    log_marginal_likelihood,
)
from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import (
    HammingKernel,
    Matern52Kernel,
    NumericKernel,
)
from neps.optimizers.bayesian_optimization.kernels.weisfilerlehman import (
    WeisfilerLehman,
)
from neps.search_spaces import SearchSpace
from neps.search_spaces.encoding import (
    CategoricalToIntegerTransformer,
    DataPack,
    MinMaxNormalizer,
    OneHotEncoder,
    TensorTransformer,
    Transformer,
    WLInputTransformer,
)
from neps.search_spaces.hyperparameters import FloatParameter, IntegerParameter

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger(__name__)


# The optimization we do for the noise is relatively cheap while the matrices
NOISE_VARIANCE_GRID = (1e-6, 1e-4, 1e-2, 1, 1e1, 1e2)


@dataclass
class ComprehensiveGP:
    space: SearchSpace
    kernels: dict[str, tuple[Sequence[str], Kernel]]

    combined_kernel: Literal["sum", "product"] = "sum"
    noise_variance: Sequence[float] = NOISE_VARIANCE_GRID
    kernel_parameter_grid: Mapping[str, Sequence[Mapping[str, Any]]] | bool = True

    optimizer: Literal["adam", "sgd"] = "adam"
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"lr": 0.1})
    optimizer_iters: int = 20
    device: torch.device | None = None

    # Post fit attributes
    K_inv_: torch.Tensor | None = None
    likelihood_: float | None = None
    y_: torch.Tensor | None = None
    y_normalized_: torch.Tensor | None = None
    y_mean_: float | None = None
    y_std_: float | None = None
    opt_kernels_: dict[str, tuple[Sequence[str], Kernel]] | None = None
    train_x_: DataPack | None = None

    def __post_init__(self):
        # TODO: Remove when search space is just definition and does not hold values.
        self.space = self.space.clone()

    def fit(
        self,
        *,
        x: DataPack,
        train_y: torch.Tensor,
    ) -> None:
        # Preprocessing
        y_ = torch.as_tensor(train_y, device=self.device, dtype=torch.float64)

        # TODO: Dunno if I like this silent hack, setting std to 1 if no std
        self.y_std_ = s if (s := torch.std(y_).item()) != 0 else 1
        self.y_mean_ = torch.mean(y_).item()
        self.y_normalized_ = (y_ - self.y_mean_) / self.y_std_
        self.y_ = y_

        # optimized kernel parameters + noise variance
        optim_vars: list[nn.Parameter] = []
        opt_kernels: dict[str, tuple[Sequence[str], Kernel]] = {}

        N: int
        for _kernel_name, (hps, kernel) in self.kernels.items():
            data = x.select(hps)
            opt_kernel, _ = kernel.grid_search(
                x=data,  # type: ignore
                y=self.y_normalized_,
            )
            optim_vars.extend(opt_kernel.parameters())
            opt_kernels[_kernel_name] = (hps, opt_kernel)

        # Now that we've optimized the kernels, we convert go convert their
        # parameters into a tensor we can further refine with some optimizer iterations
        # - Optimize kernel-lengthscales, kernel-outputscale, noise-variance
        #   and any additional parameters they wish to advertise.
        noise_variance = nn.Parameter(
            torch.tensor(1e-3, device=self.device, dtype=torch.float64)
        )
        optim_vars.append(noise_variance)

        if self.optimizer == "adam":
            optim = Adam(optim_vars, **self.optimizer_kwargs)  # type: ignore
        elif self.optimizer == "sgd":
            optim = SGD(optim_vars, **self.optimizer_kwargs)  # type: ignore
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

        K_inv: torch.Tensor | None = None
        _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
        N = len(x)
        K = _init((N, N), device=self.device, dtype=torch.float64)
        for _i in range(self.optimizer_iters):
            optim.zero_grad()

            for _kernel_name, (hps, opt_kernel) in opt_kernels.items():
                data = x.select(hps)
                k = opt_kernel.forward(data)
                K.add_(k) if self.combined_kernel == "sum" else K.mul_(k)

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
        self.opt_kernels_ = opt_kernels
        self.train_x_ = x

    def predict(
        self,
        *,
        x: DataPack,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kriging predictions."""
        if (
            self.K_inv_ is None
            or self.train_x_ is None
            or self.y_normalized_ is None
            or self.y_std_ is None
            or self.opt_kernels_ is None
        ):
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )

        _init = torch.zeros if self.combined_kernel == "sum" else torch.ones
        n_test = len(x)

        K_train_test = _init(
            len(self.train_x_), n_test, device=self.device, dtype=torch.float64
        )
        for _kernel_name, (hps, opt_kernel) in self.opt_kernels_.items():
            train = self.train_x_.select(hps)
            test = x.select(hps)
            k = opt_kernel.forward(train, test)
            if self.combined_kernel == "sum":
                K_train_test.add_(k)
            else:
                K_train_test.mul_(k)

        K_test_test = _init(n_test, n_test, device=self.device, dtype=torch.float64)
        for _kernel_name, (hps, opt_kernel) in self.opt_kernels_.items():
            test = x.select(hps)
            k = opt_kernel.forward(test, test)
            if self.combined_kernel == "sum":
                K_test_test.add_(k)
            else:
                K_test_test.mul_(k)

        # Compute the predictive mean

        # Scale by the standard deviation and mean
        mu_s = K_train_test.t() @ self.K_inv_ @ self.y_normalized_
        mu_s = mu_s * self.y_std_ + self.y_mean_

        cov_s = K_test_test - K_train_test.t() @ self.K_inv_ @ K_train_test
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
            transformer = CategoricalToIntegerTransformer(tuple(space.categoricals))
            kernels["categorical"] = (HammingKernel(), transformer)

    return kernels
