"""Samplers for generating points in a search space.

These are similar to [`Prior`][neps.priors.Prior] objects, but they
do not necessarily have an easily definable pdf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Protocol, Sequence
from typing_extensions import override

import torch
from more_itertools import all_equal

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


@dataclass
class WeightedSampler(Sampler):
    """A sampler that samples from a weighted combination of samplers."""

    samplers: Sequence[Sampler]
    """The samplers to sample from."""

    weights: torch.Tensor
    """The weights for each sampler."""

    probabilities: torch.Tensor = field(init=False, repr=False)
    """The probabilities for each sampler. Normalized weights."""

    def __post_init__(self):
        if len(self.samplers) < 2:
            raise ValueError(
                f"At least two samplers must be given. Got {len(self.samplers)}"
            )

        if self.weights.ndim != 1:
            raise ValueError("Weights must be a 1D tensor.")

        if len(self.samplers) != len(self.weights):
            raise ValueError("The number of samplers and weights must be the same.")

        ncols = [sampler.ncols for sampler in self.samplers]
        if not all_equal(ncols):
            raise ValueError(
                "All samplers must have the same number of columns." f" Got {ncols}."
            )

        self._ncols = ncols[0]
        self.probabilities = self.weights / self.weights.sum()

    @property
    @override
    def ncols(self) -> int:
        return self._ncols

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
            raise NotImplementedError("Seeding is not yet implemented.")

        # Calculate the total number of samples required
        if isinstance(n, int):
            total_samples = n
            output_shape = (n, self.ncols)
        else:
            total_samples = reduce(lambda x, y: x * y, n)
            output_shape = (*n, self.ncols)

        # Randomly select which prior to sample from for each of the total_samples
        chosen_priors = torch.empty((total_samples,), device=device, dtype=torch.int64)
        chosen_priors = torch.multinomial(
            self.probabilities,
            total_samples,
            replacement=True,
            out=chosen_priors,
        )

        # Create an empty tensor to hold all samples
        output_samples = torch.empty(
            (total_samples, self.ncols), device=device, dtype=torch.float64
        )

        # Loop through each prior and its associated indices
        for i, prior in enumerate(self.samplers):
            # Find indices where the chosen prior is i
            _i = torch.tensor(i, dtype=torch.int64, device=device)
            indices = torch.where(chosen_priors == _i)[0]

            if len(indices) > 0:
                # Sample from the prior for the required number of indices
                samples_from_prior = prior.sample(len(indices), to=to, device=device)
                output_samples[indices] = samples_from_prior

        # Reshape to the output shape including ncols dimension
        output_samples = output_samples.view(output_shape)

        return Domain.translate(output_samples, frm=UNIT_FLOAT_DOMAIN, to=to)
