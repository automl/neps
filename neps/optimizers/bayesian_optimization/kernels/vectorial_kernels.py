from __future__ import annotations
from copy import deepcopy
from math import sqrt
from dataclasses import dataclass, field
from typing_extensions import override

import numpy as np
import torch

LENGTHSCALE_BOUNDS_DEFAULT = (
    np.exp(-6.754111155189306),
    np.exp(0.0858637988771976),
)


@dataclass
class Stationary:
    """Here we follow the structure of GPy to build a sub class of stationary kernel.
    All the classes (i.e. the class of stationary kernel_operators) derived from this
    class use the scaled distance to compute the Gram matrix."""

    # A single value applies to all dimensions, a vector applies to each dimension
    lengthscale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    lengthscale_bounds: tuple[float, float] = LENGTHSCALE_BOUNDS_DEFAULT
    outputscale: float = 1.0

    gram_: torch.Tensor | None = None
    train_: torch.Tensor | None = None

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        return _scaled_distance(lengthscale, x1, x2)

    def fit_transform(
        self,
        x1,
        l: torch.Tensor | None = None,
        rebuild_model: bool = True,
        save_gram_matrix: bool = True,
    ) -> torch.Tensor:
        if not rebuild_model and self.gram_ is not None:
            return self.gram_
        K = self.forward(x1, l=l)
        if save_gram_matrix:
            self.train_ = deepcopy(x1)
            assert isinstance(K, torch.Tensor), "it doesnt work with np arrays.."
            self.gram_ = K.clone()
        return K

    def transform(self, x1, l: torch.Tensor | None = None) -> torch.Tensor:
        if self.gram_ is None or self.train_ is None:
            raise ValueError("The kernel has not been fitted. Run fit_transform first")
        return self.forward(self.train_, x1, l=l)

    def forward_t(
        self,
        x2: torch.Tensor,
        x1: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x1 is None:
            x1 = torch.tensor(self.train_)
        x2 = torch.tensor(x2, requires_grad=True)
        K = self.forward(x1, x2, l)
        return K, x2

    def update_hyperparameters(self, lengthscale: torch.Tensor) -> None:
        self.lengthscale = torch.clamp(lengthscale, *self.lengthscale_bounds)


@dataclass
class RBFKernel(Stationary):
    @override
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        dist_sq = _scaled_distance(lengthscale, x1, x2, sq_dist=True)
        return self.outputscale * torch.exp(-0.5 * dist_sq)


@dataclass
class LayeredRBFKernel(RBFKernel):
    """
    Same as the conventional RBF kernel, but adapted in a way as a midway between
    spherical RBF and ARD RBF. In this case, one weight is assigned to each
    Weisfiler-Lehman iteration only (e.g. one weight for h=0, another for h=1 and etc.)
    """

    @override
    def forward(
        self,
        ard_dims: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _l = l if l is not None else self.lengthscale
        assert isinstance(_l, torch.Tensor), "Lengthscale must be a torch tensor"
        assert _l.shape[0] == ard_dims.shape[0], (
            "LayeredRBF expects the lengthscale vector to have the same "
            "dimensionality as the "
            "number of WL iterations, but got lengthscale vector of shape"
            + str(_l.shape[0])
            + "and WL iteration of shape "
            + str(ard_dims.shape[0])
        )

        M = torch.cat(
            [torch.ones(int(ard_dims[i])) * _l[i] for i in range(len(ard_dims))]
        )
        return super().forward(x1, x2, M)


@dataclass
class Matern32Kernel(Stationary):
    @override
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        dist = _scaled_distance(lengthscale, x1, x2)
        return self.outputscale * (1 + sqrt(3.0) * dist) * torch.exp(-sqrt(3.0) * dist)


class Matern52Kernel(Stationary):
    @override
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        dist = _scaled_distance(lengthscale, x1, x2, sq_dist=True)
        return (
            self.outputscale
            * (1 + sqrt(5.0) * dist + 5.0 / 3.0 * dist)
            * torch.exp(-sqrt(5.0) * dist)
        )


@dataclass
class HammingKernel(Stationary):
    @override
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        dist = _hamming_distance(lengthscale, x1, x2)
        return self.outputscale * dist


@dataclass
class RationalQuadraticKernel(Stationary):
    power: float = 2.0

    @override
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        l: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengthscale = l if l is not None else self.lengthscale
        dist_sq = _scaled_distance(lengthscale, x1, x2, sq_dist=True)
        return self.outputscale * (1 + dist_sq / 2.0) ** (-self.power)


def _unscaled_square_distance(
    X: torch.Tensor,
    X2: torch.Tensor | None = None,
) -> torch.Tensor:
    """The unscaled distance between X and X2."""
    assert X.ndim == 2
    X1sq = torch.sum(X**2, 1)
    X2sq = X1sq if X is X2 else torch.sum(X**2, 1)
    X2 = X if X2 is None else X2

    r2 = -2 * X @ X2.T + X1sq[:, None] + X2sq[None, :]
    r2 += 1e-15
    return torch.clamp_min(r2, 0.0)


def _scaled_distance(
    lengthscale: torch.Tensor,
    X: torch.Tensor,
    X2: torch.Tensor | None = None,
    *,
    sq_dist: bool = False,
) -> torch.Tensor:
    """Compute the *scaled* distance between X and x2 (or, if X2 is not supplied,
    the distance between X and itself) by the lengthscale. if a scalar (float) or a
    dim=1 lengthscale vector is supplied, then it is assumed that we use one
    lengthscale for all dimensions. Otherwise, we have an ARD kernel and in which case
    the length of the lengthscale vector must be the same as the dimensionality of the
    problem."""
    if len(lengthscale) == 1:
        if sq_dist is False:
            return torch.sqrt(_unscaled_square_distance(X, X2)) / (lengthscale**2)

        return _unscaled_square_distance(X, X2) / lengthscale

    # ARD kernel - one lengthscale per dimension
    assert len(lengthscale) == X.shape[1], (
        f"Lengthscale must have the same dimensionality as the input data."
        f"Got {len(lengthscale)} and {X.shape[1]}"
    )
    rescaled_X = X / lengthscale
    if X2 is None:
        dist = _unscaled_square_distance(rescaled_X)
    else:
        rescaled_X2 = X2 / lengthscale
        dist = _unscaled_square_distance(rescaled_X, rescaled_X2)

    return dist if sq_dist else torch.sqrt(dist)


def _hamming_distance(
    lengthscale: torch.Tensor,
    X: torch.Tensor,
    X2: torch.Tensor | None = None,
) -> torch.Tensor:
    if X2 is None:
        X2 = X

    indicator = X.unsqueeze(1) != X2
    C = -1 / (2 * lengthscale**2)
    scaled_indicator = C * indicator
    diffs = scaled_indicator.sum(dim=2)

    if len(lengthscale) == 1:
        return torch.exp(diffs) / lengthscale

    return torch.exp(diffs)
