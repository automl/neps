"""Samplers for generating points in a search space.

These are similar to [`Prior`][scalefm.sampling.Prior] objects, but they
do not necessarily have an easily definable pdf.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch

from scalefm.space import ConfigEncoder, Domain

if TYPE_CHECKING:
    from scalefm.sampling.priors import Uniform


class Sampler(ABC):
    """A protocol for sampling tensors and vonerting them to a given domain."""

    @property
    @abstractmethod
    def ncols(self) -> int:
        """The number of columns in the samples produced by this sampler."""
        ...

    @abstractmethod
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain] | ConfigEncoder,
        seed: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample `n` points and convert them to the given domain.

        Args:
            n: The number of points to sample. If a torch.Size, an additional dimension
                will be added with [`.ncols`][scalefm.sampling.Sampler.ncols].
                For example, if `n = 5`, the output will be `(5, ncols)`. If
                `n = (5, 3)`, the output will be `(5, 3, ncols)`.
            to: If a single domain, `.ncols` columns will be produced form that one
                domain. If a list of domains, then it must have the same length as the
                number of columns, with each column being in the corresponding domain.
            seed: The seed generator
            dtype: The dtype of the output tensor.
            device: The device to cast the samples to.

        Returns:
            A tensor of (n, ndim) points sampled cast to the given domain.
        """

    def sample_config(
        self,
        to: ConfigEncoder,
        *,
        seed: torch.Generator | None = None,
        include: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """See [`sample_configs()`][scalefm.sampling.Sampler.sample_configs]."""
        return self.sample_configs(1, to, seed=seed, include=include)[0]

    def sample_configs(
        self,
        n: int,
        to: ConfigEncoder,
        *,
        seed: torch.Generator | None = None,
        include: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Sample configurations directly into a search space.

        Args:
            n: The number of configurations to sample.
            to: The encoding to sample into.
            seed: The seed generator.
            include: Additional values to include in the configuration.

        Returns:
            A list of configurations.
        """
        tensors = self.sample(n, to=to, seed=seed)
        configs = to.decode(tensors)
        if include is None:
            return configs
        return [{**config, **include} for config in configs]

    @classmethod
    def sobol(cls, ndim: int, *, scramble: bool = True) -> Sobol:
        """Create a Sobol sampler.

        Args:
            ndim: The number of columns to sample.
            scramble: Whether to scramble the Sobol sequence.

        Returns:
            A Sobol sampler.
        """
        return Sobol(ndim=ndim, scramble=scramble)

    @classmethod
    def uniform(cls, ndim: int) -> Uniform:
        """Create a uniform sampler.

        Args:
            ndim: The number of columns to sample.

        Returns:
            A uniform sampler.
        """
        from scalefm.sampling.priors import Uniform

        return Uniform(ndim=ndim)


# Technically this could be a prior with a uniform distribution
@dataclass
class Sobol(Sampler):
    """Sample from a Sobol sequence."""

    ndim: int
    """The number of dimensions to sample for."""

    scramble: bool = True
    """Whether to scramble the Sobol sequence."""

    def __post_init__(self) -> None:
        if self.ndim < 1:
            raise ValueError(
                "The number of dimensions must be at least 1."
                f" Got {self.ndim} dimensions."
            )

    @property
    @override
    def ncols(self) -> int:
        return self.ndim

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain] | ConfigEncoder,
        seed: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Setting the seed is not supported yet")

        # Sobol can only produce 2d tensors. To handle batches or arbitrary
        # dimensions, we get a count of the total number of samples needed
        # and reshape the output tensor to the desired shape, if needed.
        _n = n if isinstance(n, int) else reduce(lambda x, y: x * y, n)

        _seed = (
            None if seed is None else torch.randint(0, 2**31, (1,), generator=seed).item()
        )
        sobol = torch.quasirandom.SobolEngine(
            dimension=self.ndim, scramble=self.scramble, seed=_seed
        )

        # If integer dtype, sobol will refuse, we need to cast then
        if dtype is not None and not dtype.is_floating_point:
            x = sobol.draw(_n, dtype=torch.float64)
            x = x.to(dtype=dtype, device=device)
        else:
            x = sobol.draw(_n, dtype=dtype)

        # If we got extra dimensions, such as batch dimensions, we need to
        # reshape the tensor to the desired shape.
        if isinstance(n, torch.Size):
            x = x.view(*n, self.ncols)

        return Domain.translate(x, frm=Domain.unit_float(), to=to)
