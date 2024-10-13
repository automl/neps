"""Samplers for generating points in a search space.

These are similar to [`Prior`][neps.priors.Prior] objects, but they
do not necessarily have an easily definable pdf.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Protocol
from typing_extensions import override

import torch
from more_itertools import all_equal

from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from neps.sampling.priors import UniformPrior


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
        to: Domain | list[Domain] | ConfigEncoder,
        seed: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample `n` points and convert them to the given domain.

        Args:
            n: The number of points to sample. If a torch.Size, an additional dimension
                will be added with [`.ncols`][neps.samplers.Sampler.ncols].
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
        ...

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
    def uniform(cls, ndim: int) -> UniformPrior:
        """Create a uniform sampler.

        Args:
            ndim: The number of columns to sample.

        Returns:
            A uniform sampler.
        """
        from neps.sampling.priors import UniformPrior

        return UniformPrior(ndim=ndim)

    @classmethod
    def borders(cls, ndim: int) -> BorderSampler:
        """Create a border sampler.

        Args:
            ndim: The number of dimensions to sample.

        Returns:
            A border sampler.
        """
        return BorderSampler(ndim=ndim)


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

        return Domain.translate(x, frm=UNIT_FLOAT_DOMAIN, to=to)


@dataclass
class WeightedSampler(Sampler):
    """A sampler that samples from a weighted combination of samplers."""

    samplers: Sequence[Sampler]
    """The samplers to sample from."""

    weights: torch.Tensor
    """The weights for each sampler."""

    sampler_probabilities: torch.Tensor = field(init=False, repr=False)
    """The probabilities for each sampler. Normalized weights."""

    def __post_init__(self) -> None:
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
        self.sampler_probabilities = self.weights / self.weights.sum()

    @property
    @override
    def ncols(self) -> int:
        return self._ncols

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
        if dtype is None:
            match to:
                case Domain():
                    dtype = to.preffered_dtype
                case ConfigEncoder():
                    dtype = (
                        torch.float64
                        if any(d.preffered_dtype.is_floating_point for d in to.domains)
                        else torch.int64
                    )
                case _:
                    dtype = (
                        torch.float64
                        if any(d.preffered_dtype.is_floating_point for d in to)
                        else torch.int64
                    )

        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        # Calculate the total number of samples required
        if isinstance(n, int):
            total_samples = n
            output_shape = (n, self.ncols)
        else:
            total_samples = reduce(lambda x, y: x * y, n)
            output_shape = (*n, self.ncols)

        # Randomly select which sampler to sample from for each of the total_samples
        chosen_samplers = torch.empty((total_samples,), device=device, dtype=torch.int64)
        chosen_samplers = torch.multinomial(
            self.sampler_probabilities,
            total_samples,
            replacement=True,
            generator=seed,
            out=chosen_samplers,
        )

        # Create an empty tensor to hold all samples
        output_samples = torch.empty(
            (total_samples, self.ncols),
            device=device,
            dtype=dtype,
        )

        # Loop through each sampler and its associated indices
        for i, sampler in enumerate(self.samplers):
            # Find indices where the chosen sampler is i
            _i = torch.tensor(i, dtype=torch.int64, device=device)
            indices = torch.where(chosen_samplers == _i)[0]

            if len(indices) > 0:
                # Sample from the sampler for the required number of indices
                samples_from_sampler = sampler.sample(
                    len(indices),
                    to=to,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                )
                output_samples[indices] = samples_from_sampler

        # Reshape to the output shape including ncols dimension
        return output_samples.view(output_shape)


@dataclass
class BorderSampler(Sampler):
    """A sampler that samples from the border of a hypercube."""

    ndim: int

    @property
    @override
    def ncols(self) -> int:
        return self.ndim

    @property
    def n_possible(self) -> int:
        """The amount of possible border configurations."""
        return 2**self.ndim

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
        dtype = dtype or torch.float64

        _arange = torch.arange(self.n_possible, device=device, dtype=torch.int32)
        # Calculate the total number of samples required
        if isinstance(n, int):
            total_samples = min(n, self.n_possible)
            output_shape = (total_samples, self.ncols)
        else:
            total_samples = reduce(lambda x, y: x * y, n)
            if total_samples > self.n_possible:
                raise ValueError(
                    f"The shape of samples requested (={n}) is more than the number of "
                    f"possible border configurations (={self.n_possible})."
                )
            output_shape = (*n, self.ncols)

        if self.n_possible <= total_samples:
            configs = _arange
        else:
            # Otherwise, we take a random sample of the 2**n possible border configs
            rand_ix = torch.randperm(self.n_possible, generator=seed, device=device)[
                :total_samples
            ]
            configs = _arange[rand_ix]

        # https://stackoverflow.com/a/63546308/5332072
        bit_masks = 2 ** _arange[: self.ndim]
        configs = configs.unsqueeze(1).bitwise_and(bit_masks).ne(0).to(dtype)
        # Reshape to the output shape including ncols dimension
        configs = configs.view(output_shape)
        return Domain.translate(configs, frm=UNIT_FLOAT_DOMAIN, to=to)
