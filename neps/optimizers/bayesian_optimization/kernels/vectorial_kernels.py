from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence, ClassVar
from typing_extensions import override, Self

from itertools import product
import torch
import torch.nn as nn

from neps.optimizers.bayesian_optimization.kernels.kernel import Kernel

# TODO:
# We should try some variations of singular length scales
# (1 scale shared across all dimensions)
# and individual ARD lengthscales (1 for each dimension)
# ARD can overfit if not properly tuned...
LENGTHSCALE_GRID = (1e-2, 1e-1, 1, 1e1, 1e2)
STD_ENCODED_OUTPUT_SCALE = (1e-2, 1e-1, 1, 1e1, 1e2)


class Stationary(Kernel[torch.Tensor]):
    suggested_grid: ClassVar[Sequence[Mapping[str, Any]]] = [
        {"lengthscale": l, "output_scale": o}
        for l, o in product(LENGTHSCALE_GRID, STD_ENCODED_OUTPUT_SCALE)
    ]

    def __init__(
        self,
        *,
        lengthscale: torch.Tensor | None = None,
        outputscale: torch.Tensor | None = None,
        lengthscale_bounds: tuple[float, float] | None = (1e-2, 1e2),
        outputscale_bounds: tuple[float, float] | None = (1e-2, 1e2),
        device: torch.device | None = None,
    ):
        super().__init__()
        self.lengthscale = (
            torch.as_tensor(lengthscale, dtype=torch.float64, device=device)
            if lengthscale is not None
            else torch.tensor(1, dtype=torch.float64, device=device)
        )
        self.outputscale = (
            torch.as_tensor(outputscale, dtype=torch.float64, device=device)
            if outputscale is not None
            else torch.tensor(1, dtype=torch.float64, device=device)
        )
        self.lengthscale_bounds = lengthscale_bounds
        self.outputscale_bounds = outputscale_bounds
        self.device = device

        self.train_: torch.Tensor | None = None

    def as_optimizable(self) -> Self:
        return self.clone_with(
            lengthscale=nn.Parameter(self.lengthscale),
            outputscale=nn.Parameter(self.outputscale),
        )

    def forward(self, x: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        # NOTE: I don't think this is the right way to do this...
        with torch.no_grad():
            self.lengthscale.data.clamp_(*self.lengthscale_bounds)
            self.outputscale.data.clamp_(*self.outputscale_bounds)

        x2 = x if x2 is None else x2
        return self._forward(x, x2)

    def _forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.outputscale * torch.cdist(x1, x2, p=2)


class RBFKernel(Stationary):
    @override
    def _forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(x1, x2, p=2) ** 2
        return self.outputscale * torch.exp(-dist_sq / (2 * self.lengthscale**2))


class Matern32Kernel(Stationary):
    @override
    def _forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1, x2, p=2) / self.lengthscale
        factor = sqrt(3.0) * dist
        matern32 = (1 + factor) * torch.exp(-factor)
        return self.outputscale * matern32


class HammingKernel(Stationary):
    @override
    def _forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dists = (x1.unsqueeze(1) != x2.unsqueeze(0)).float().sum(-1) / x1.shape[-1]
        scaled_dists = dists / self.lengthscale
        return self.outputscale * torch.exp(-scaled_dists)


class Matern52Kernel(Stationary):
    @override
    def _forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1, x2, p=2) / self.lengthscale
        factor = sqrt(5.0) * dist
        matern52 = (1 + factor + (factor**2) / 3) * torch.exp(-factor)
        return self.outputscale * matern52
