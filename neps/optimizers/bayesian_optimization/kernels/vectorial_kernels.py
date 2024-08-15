from __future__ import annotations

from math import sqrt
from typing_extensions import override
from neps.optimizers.bayesian_optimization.kernels.kernel import Kernel

import numpy as np
import torch

DEFAULT_LENGTHSCALE_BOUNDS = np.exp(-6.754111155189306), np.exp(0.0858637988771976)


class Stationary(Kernel[torch.Tensor]):
    """Here we follow the structure of GPy to build a sub class of stationary kernel.

    All the classes (i.e. the class of stationary kernel_operators) derived from this
    class use the scaled distance to compute the Gram matrix.
    """

    def __init__(
        self,
        *,
        lengthscale: torch.Tensor,
        outputscale: float | torch.Tensor = 1.0,
        lengthscale_bounds: tuple[float, float] = DEFAULT_LENGTHSCALE_BOUNDS,
    ):
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.lengthscale_bounds = lengthscale_bounds

        self.gram_: torch.Tensor | None = None
        self.train_: torch.Tensor | None = None

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        K = self._forward(x)
        self.train_ = x.clone().detach()
        return K

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_ is None:
            raise ValueError("The kernel has not been fitted. Run fit_transform first")
        return self._forward(self.train_, x)

    def _forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        return _scaled_distance(self.lengthscale, x1, x2)


class RBFKernel(Stationary):
    @override
    def _forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist_sq = _scaled_distance(self.lengthscale, x1, x2, sq_dist=True)
        return self.outputscale * torch.exp(-0.5 * dist_sq)


class Matern32Kernel(Stationary):
    @override
    def _forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = _scaled_distance(self.lengthscale, x1, x2)
        return self.outputscale * (1 + sqrt(3.0) * dist) * torch.exp(-sqrt(3.0) * dist)


class Matern52Kernel(Stationary):
    @override
    def _forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = _scaled_distance(self.lengthscale, x1, x2, sq_dist=True)
        return (
            self.outputscale
            * (1 + sqrt(5.0) * dist + 5.0 / 3.0 * dist)
            * torch.exp(-sqrt(5.0) * dist)
        )


def _unscaled_square_distance(
    X: torch.Tensor,
    X2: torch.Tensor | None = None,
) -> torch.Tensor:
    """The unscaled distance between X and X2."""
    assert X.ndim == 2
    X1sq = torch.sum(X**2, 1)
    X2sq = X1sq if (X2 is None or X is X2) else torch.sum(X2**2, 1)
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
