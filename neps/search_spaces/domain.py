"""A class representing a domain, a range for a value + properties.

Some properties include:

* The lower and upper bounds of the domain.
* Whether the domain is a log domain.
* Whether the domain is float/int.
* The midpoint of the domain.
* Whether the domain is split into bins.

With that, the primary method of a domain is to be able to
[`cast()`][neps.search_spaces.domain.Domain.cast] a tensor of
values from one to domain to another,
e.g. `values_a = domain_a.cast(values_b, frm=domain_b)`.

This can be used to convert float samples to integers, integers
to log space, etc.

The core method to do so is to be able to cast
[`to_unit()`][neps.search_spaces.domain.Domain.to_unit] which takes
values to a unit interval [0, 1], and then to be able to cast values in [0, 1]
to the new domain with [`from_unit()`][neps.search_spaces.domain.Domain.from_unit].

There are some shortcuts implemented in `cast`, such as skipping going through
the unit interval if the domains are the same, as no transformation is needed.

The primary methods for creating a domain are

* [`Domain.float(l, u, ...)`][neps.search_spaces.domain.Domain.float] -
    Used for modelling float ranges
* [`Domain.int(l, u, ...)`][neps.search_spaces.domain.Domain.int] -
    Used for modelling integer ranges
* [`Domain.indices(n)`][neps.search_spaces.domain.Domain.indices] -
    Primarly used to model categorical choices

If you have a tensor of values, where each column corresponds to a different domain,
you can take a look at [`Domain.translate()`][neps.search_spaces.domain.Domain.translate]

If you need a unit-interval domain, please use the
[`Domain.unit_float()`][neps.search_spaces.domain.Domain.unit_float]
or `UNIT_FLOAT_DOMAIN` constant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
from torch import Tensor

Number = int | float
V = TypeVar("V", int, float)
V2 = TypeVar("V2", int, float)


@dataclass(frozen=True)
class Domain(Generic[V]):
    """A domain for a value.

    The primary methods for creating a domain are

    * [`Domain.float(l, u, ...)`][neps.search_spaces.domain.Domain.float] -
        Used for modelling float ranges
    * [`Domain.int(l, u, ...)`][neps.search_spaces.domain.Domain.int] -
        Used for modelling integer ranges
    * [`Domain.indices(n)`][neps.search_spaces.domain.Domain.indices] -
        Primarly used to model categorical choices
    """

    lower: V
    """The lower bound of the domain."""

    upper: V
    """The upper bound of the domain."""

    round: bool
    """Whether to round the values to the nearest integer."""

    log_bounds: tuple[float, float] | None = None
    """The log bounds of the domain, if the domain is in log space."""

    bins: int | None = None
    """The number of discrete bins to split the domain into.

    Includes both endpoints of the domain and values are rounded to the nearest bin
    value.
    """

    dtype: torch.dtype = field(init=False, repr=False)
    is_unit_float: bool = field(init=False, repr=False)
    midpoint: V = field(init=False, repr=False)
    is_log: bool = field(init=False, repr=False)
    length: V = field(init=False, repr=False)
    cardinality: int | None = field(init=False, repr=False)
    bounds: tuple[V, V] = field(init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.lower, type(self.upper))
        is_int = isinstance(self.lower, int)
        object.__setattr__(self, "is_log", self.log_bounds is not None)
        object.__setattr__(self, "dtype", torch.int64 if is_int else torch.float64)
        object.__setattr__(
            self,
            "is_unit_float",
            self.lower == 0 and self.upper == 1 and is_int and not self.round,
        )
        object.__setattr__(self, "length", self.upper - self.lower)

        if self.bins:
            cardinality = self.bins
        elif self.round:
            cardinality = int(self.upper - self.lower + 1)
        else:
            cardinality = None

        object.__setattr__(self, "cardinality", cardinality)
        mid = self.from_unit(torch.tensor(0.5)).item()
        if self.dtype == torch.int64:
            mid = int(round(mid))
        object.__setattr__(self, "midpoint", mid)
        object.__setattr__(self, "bounds", (self.lower, self.upper))

    @classmethod
    def float(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
    ) -> Domain[float]:
        """Create a domain for a range of float values.

        Args:
            lower: The lower bound of the domain.
            upper: The upper bound of the domain.
            log: Whether the domain is in log space.
            bins: The number of discrete bins to split the domain into.

        Returns:
            A domain for a range of float values.
        """
        return Domain(
            lower=float(lower),
            upper=float(upper),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            bins=bins,
            round=False,
        )

    @classmethod
    def int(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
    ) -> Domain[int]:
        """Create a domain for a range of integer values.

        Args:
            lower: The lower bound of the domain.
            upper: The upper bound of the domain.
            log: Whether the domain is in log space.
            bins: The number of discrete bins to split the domain into.

        Returns:
            A domain for a range of integer values.
        """
        return Domain(
            lower=int(round(lower)),
            upper=int(round(upper)),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            round=True,
            bins=bins,
        )

    def next_value(self, x: Tensor) -> Tensor:
        """Get the next value for a tensor of values."""
        if self.cardinality is None:
            raise ValueError("Domain is non-finite, cannot get next value.")
        cardinality_domain = Domain.indices(self.cardinality)
        current_step = cardinality_domain.cast(x, frm=self)
        bounded_next_step = (current_step + 1).clamp_max(self.cardinality - 1)
        return self.cast(bounded_next_step, frm=cardinality_domain)

    @classmethod
    def indices(cls, n: int) -> Domain[int]:
        """Create a domain for a range of indices.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Args:
            n: The number of indices.

        Returns:
            A domain for a range of indices.
        """
        return Domain.int(0, n - 1)

    def to_unit(self, x: Tensor) -> Tensor:
        """Transform a tensor of values from this domain to the unit interval [0, 1].

        Args:
            x: Tensor of values in this domain to convert.

        Returns:
            Same shape tensor with the values normalized to the unit interval [0, 1].
        """
        if self.is_unit_float:
            return x

        if self.log_bounds is not None:
            x = torch.log(x)
            lower, upper = self.log_bounds
        else:
            lower, upper = self.lower, self.upper

        return (x - lower) / (upper - lower)

    def from_unit(self, x: Tensor) -> Tensor:
        """Transform a tensor of values from the unit interval [0, 1] to this domain.

        Args:
            x: A tensor of values in the unit interval [0, 1] to convert.

        Returns:
            Same shape tensor with the lifted into this domain.
        """
        if self.is_unit_float:
            return x

        bins = self.bins
        if bins is not None:
            quantization_levels = torch.floor(x * bins).clip(0, bins - 1)
            x = quantization_levels / (bins - 1)

        # Now we scale to the new domain
        if self.log_bounds is not None:
            lower, upper = self.log_bounds
            x = x * (upper - lower) + lower
            x = torch.exp(x)
        else:
            lower, upper = self.lower, self.upper
            x = x * (upper - lower) + lower

        if self.round:
            x = torch.round(x)

        return x.type(self.dtype)

    def cast(self, x: Tensor, frm: Domain) -> Tensor:
        """Cast a tensor of values frm the domain `frm` to this domain.

        If you need to cast a tensor of mixed domains, use
        [`Domain.translate()`][neps.search_spaces.domain.Domain.translate].

        Args:
            x: Tensor of values in the `frm` domain to cast to this domain.
            frm: The domain to cast from.

        Returns:
            Same shape tensor with the values cast to this domain.
        """
        # NOTE: In general, we should always be able to go through the unit interval
        # [0, 1] to be able to transform between domains. However sometimes we can
        # bypass some steps, dependant on the domains, hence the ugliness...

        # Shortcut 1. (Same Domain)
        # We can shortcut out going through normalized space if all the boundaries and
        # they live on the same scale. However, if their bins don't line up, we will
        # have to go through unit space to figure out the bins
        same_bounds = self.lower == frm.lower and self.upper == frm.upper
        same_log_bounds = self.log_bounds == frm.log_bounds
        same_bins = self.bins == frm.bins
        if same_bounds and same_log_bounds and (self.bins is None or same_bins):
            if self.round:
                x = torch.round(x)
            return x.type(self.dtype) if x.dtype != self.dtype else x

        # Shortcut 2. (From normalized)
        # The domain we are coming from is already normalized, we only need to lift
        if frm.is_unit_float:
            return self.from_unit(x)  # type: ignore

        # Shortcut 3. (Log lift)
        # We can also shortcut out if the only diffrence is that we are coming frm the
        # log bounds of this domain. We dont care if where we came from was binned or not,
        # we just lift it up with `np.exp` and round if needed
        if (self.lower, self.upper) == frm.log_bounds and self.bins is None:
            x = torch.exp(x)
            if self.round:
                x = torch.round(x)
            return x.type(self.dtype)

        # Otherwise, through the unit interval we go
        norm = frm.to_unit(x)
        lift = self.from_unit(norm)
        return lift  # noqa: RET504

    @classmethod
    def unit_float(cls) -> Domain[float]:
        """Get a domain for the unit interval [0, 1]."""
        return UNIT_FLOAT_DOMAIN

    @classmethod
    def translate(
        cls,
        x: Tensor,
        frm: Domain | Iterable[Domain],
        to: Domain | Iterable[Domain],
    ) -> Tensor:
        """Cast a tensor of mixed domains to a new set of mixed domains.

        Args:
            x: Tensor of shape (..., n_dims) with each dim `i` corresponding
                to the domain `frm[i]`.
            frm: List of domains to cast from. If list, must be length of `n_dims`,
                otherwise we assume the single domain provided is the one to be used
                across all dimensions.
            to: List of domains to cast to. If list, must be length as `n_dims`,
                otherwise we assume the single domain provided is the one to be used
                across all dimensions.

        Returns:
            Tensor of the same shape as `x` with the last dimension casted
                from the domain `frm[i]` to the domain `to[i]`.
        """
        if x.ndim == 0:
            raise ValueError("Expected a tensor with at least one dimension.")

        if x.ndim == 1:
            x = x.unsqueeze(0)

        ndims = x.shape[-1]

        # If both are not a list, we can just cast the whole tensor
        if isinstance(frm, Domain) and isinstance(to, Domain):
            return to.cast(x, frm=frm)

        frm = [frm] * ndims if isinstance(frm, Domain) else list(frm)
        to = [to] * ndims if isinstance(to, Domain) else list(to)

        if len(frm) != ndims:
            raise ValueError(
                "The number of domains in `frm` must match the number of tensors"
                " if provided as a list."
                f" Expected {ndims} from last dimension of {x.shape}, got {len(frm)}."
            )

        if len(to) != ndims:
            raise ValueError(
                "The number of domains in `to` must match the number of tensors"
                " if provided as a list."
                f" Expected {ndims} from last dimension of {x.shape=}, got {len(to)}."
            )

        out = torch.empty_like(x)
        for i, (f, t) in enumerate(zip(frm, to, strict=False)):
            out[..., i] = t.cast(x[..., i], frm=f)

        return out

    def cast_one(self, x: float | int, frm: Domain) -> float | int:
        """Cast a single value from the domain `frm` to this domain.

        Args:
            x: Value in the `frm` domain to cast to this domain.
            frm: The domain to cast from.

        Returns:
            Value cast to this domain.
        """
        return self.cast(torch.tensor(x), frm=frm).item()

    def from_unit_one(self, x: float) -> float | int:
        """Transform a single value from the unit interval [0, 1] to this domain.

        Args:
            x: A value in the unit interval [0, 1] to convert.

        Returns:
            Value lifted into this domain.
        """
        return self.from_unit(torch.tensor(x)).item()

    def to_unit_one(self, x: float | int) -> float:
        """Transform a single value from this domain to the unit interval [0, 1].

        Args:
            x: Value in this domain to convert.

        Returns:
            Value normalized to the unit interval [0, 1].
        """
        return self.to_unit(torch.tensor(x)).item()


UNIT_FLOAT_DOMAIN = Domain.float(0.0, 1.0)
