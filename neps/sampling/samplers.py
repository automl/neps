"""Samplers for generating points in a search space.

These are similar to [`Prior`][neps.priors.Prior] objects, but they
do not necessarily have an easily definable pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Protocol
from typing_extensions import override

import torch

from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain


class Sampler(Protocol):
    """A protocol for sampling tensors and vonerting them to a given domain."""

    @property
    def ncols(self) -> int:
        """The number of columns in the samples produced by this sampler."""
        ...

    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain],
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample `n` points and convert them to the given domain.

        Args:
            n: The number of points to sample. If a torch.Size, an additional dimension
                will be added with [`.ncols`][neps.samplers.Sampler.ncols].
                For example, if `n = 5`, the output will be `(5, ncols)`. If
                `n = (5, 3)`, the output will be `(5, 3, ncols)`.
            to: The domain or list of domains to cast the points to.
                If a single domain, all points are cast to that domain, otherwise
                each column `ndim_i` in (n, ndim) is cast to the corresponding domain
                in `to`. As a result, the length of `to` must match the number of columns
                from [`.ncols`][neps.samplers.Sampler.ncols].
            seed: The seed for the random number generator.
            device: The device to cast the samples to.

        Returns:
            A tensor of (n, ndim) points sampled cast to the given domain.
        """
        ...

    @classmethod
    def sobol(cls, ndim: int, *, scramble: bool = True, seed: int | None = None) -> Sobol:
        """Create a Sobol sampler.

        Args:
            ndim: The number of dimensions to sample for.
            scramble: Whether to scramble the Sobol sequence.
            seed: The seed for the Sobol sequence.

        Returns:
            A Sobol sampler.
        """
        return Sobol(ndim=ndim, scramble=scramble, seed=seed)


# Technically this could be a prior with a uniform distribution
@dataclass
class Sobol(Sampler):
    """Sample from a Sobol sequence."""

    ndim: int
    """The number of dimensions to sample for."""

    seed: int | None = None
    """The seed for the Sobol sequence."""

    scramble: bool = True
    """Whether to scramble the Sobol sequence."""

    @property
    @override
    def ncols(self) -> int:
        return self.ndim

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain],
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Setting the seed is not supported yet")

        # Sobol can only produce 2d tensors. To handle batches or arbitrary
        # dimensions, we get a count of the total number of samples needed
        # and reshape the output tensor to the desired shape, if needed.
        _n = n if isinstance(n, int) else reduce(lambda x, y: x * y, n)

        sobol = torch.quasirandom.SobolEngine(
            dimension=self.ndim,
            scramble=self.scramble,
            seed=self.seed,
        )

        out = torch.empty(_n, self.ncols, dtype=torch.float64, device=device)
        x = sobol.draw(_n, dtype=torch.float64, out=out)

        # If we got extra dimensions, such as batch dimensions, we need to
        # reshape the tensor to the desired shape.
        if isinstance(n, torch.Size):
            x = x.view(*n, self.ncols)

        return Domain.translate(x, frm=UNIT_FLOAT_DOMAIN, to=to)
