from __future__ import annotations

import botorch
import gpytorch
import torch

from ....search_spaces import CategoricalParameter, FloatParameter
from ....search_spaces.parameter import HpTensorShape
from .base_kernel import CustomKernel, Kernel


class BaseNumericalKernel(Kernel):
    # pylint: disable=abstract-method
    def does_apply_on(self, hp):
        return isinstance(hp, FloatParameter)


class MaternKernel(BaseNumericalKernel):
    def __init__(self, nu: float = 2.5, **kwargs):
        """Matern kernel

        Args:
            nu: The smoothness parameter. Should be 0.5, 1.5, or 2.5.
        """
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu should be 0.5, 1.5, or 2.5 for the Matern kernel")
        self.nu = nu
        super().__init__(**kwargs)

    def build(self, hp_shapes: dict[str, HpTensorShape]):
        return self._kernel_builder(
            hp_shapes, gpytorch.kernels.MaternKernel, {"nu": self.nu}
        )


class RBFKernel(BaseNumericalKernel, CustomKernel):
    def forward(self, x1, x2, gpytorch_kernel, **kwargs):
        dist_sq = _scaled_distance(gpytorch_kernel.lengthscale, x1, x2, sq_dist=True)
        return torch.exp(-0.5 * dist_sq)


class LayeredRBFKernel(RBFKernel):
    """
    Same as the conventional RBF kernel, but adapted in a way as a midway between
    spherical RBF and ARD RBF. In this case, one weight is assigned to each
    Weisfiler-Lehman iteration only (e.g. one weight for h=0, another for h=1 and etc.)
    """

    def __init__(self, ard_dims: list, **kwargs):
        super().__init__(**kwargs)
        self.ard_dims = ard_dims

    def forward(self, x1, x2, gpytorch_kernel, **kwargs):
        assert gpytorch_kernel.lengthscale.shape[0] == len(self.ard_dims), (
            "LayeredRBF expects the lengthscale vector to have the same "
            "dimensionality as the number of WL iterations, "
            f"but got lengthscale vector of shape {gpytorch_kernel.lengthscale.shape[0]}"
            f"and WL iteration of shape {len(self.ard_dims)}"
        )
        L = torch.cat(
            [
                torch.ones(self.ard_dims[i]) * gpytorch_kernel.lengthscale[i]
                for i in range(len(self.ard_dims))
            ]
        )
        dist_sq = _scaled_distance(L, x1, x2, sq_dist=True)
        return torch.exp(-0.5 * dist_sq)


class RationalQuadraticKernel(BaseNumericalKernel, CustomKernel):
    def __init__(self, power=2.0, **kwargs):
        super().__init__(**kwargs)
        self.power = power

    def forward(self, x1, x2, gpytorch_kernel, **kwargs):
        dist_sq = _scaled_distance(gpytorch_kernel.lengthscale, x1, x2, sq_dist=True)
        return (1 + dist_sq / 2.0) ** (-self.power)


class BaseCategoricalKernel(Kernel):
    # pylint: disable=abstract-method
    def does_apply_on(self, hp):
        return isinstance(hp, CategoricalParameter)


class CategoricalBotorchKernel(BaseCategoricalKernel):
    def build(self, hp_shapes: dict[str, HpTensorShape]):
        return self._kernel_builder(
            hp_shapes,
            botorch.models.kernels.CategoricalKernel,
        )


class HammingKernel(BaseCategoricalKernel, CustomKernel):
    def forward(self, x1, x2, gpytorch_kernel, **kwargs):
        indicator = x1.unsqueeze(1) != x2
        K = (-1 / (2 * gpytorch_kernel.lengthscale**2) * indicator).mean(axis=-1)
        return torch.exp(K)


def _unscaled_distance(X, X2=None, sq_dist=False):
    """The unscaled squared distance between X and X2"""
    assert X.ndimension() == X2.ndimension() == 2
    X1sq = torch.sum(X**2, axis=-1)
    X2sq = torch.sum(X2**2, axis=-1)
    r2 = -2 * X @ X2.t() + X1sq[:, None] + X2sq[None, :]
    r2 = torch.maximum(r2 + 1e-8, torch.tensor(0))
    return r2 if sq_dist else torch.sqrt(r2)


def _scaled_distance(lengthscale, X, X2=None, sq_dist=False):
    """Compute the *scaled* distance between X and x2 (or, if X2 is not supplied,
    the distance between X and itself) by the lengthscale. if the
    lengthscale shape is (1,), then it is assumed that we use one
    lengthscale for all dimensions. Otherwise, we have an ARD kernel and in which case
    the length of the lengthscale vector must be the same as the dimensionality of the
    problem."""
    if X2 is None:
        X2 = X
    if lengthscale.nelement() == 1:
        dist = _unscaled_distance(X, X2, sq_dist)
        dist /= lengthscale**2 if sq_dist else lengthscale
    else:
        # ARD kernel - one lengthscale per dimension
        dist = _unscaled_distance(X / lengthscale, X2 / lengthscale, sq_dist)
    return dist
