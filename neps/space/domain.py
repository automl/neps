"""A class representing a domain, a range for a value + properties.

Some properties include:

* The lower and upper bounds of the domain.
* Whether the domain is a log domain.
* Whether the domain is float/int.
* Whether the domain is split into bins.

With that, the primary method of a domain is to be able to
[`cast()`][neps.space.domain.Domain.cast] a tensor of
values from one to domain to another,
e.g. `values_a = domain_a.cast(values_b, frm=domain_b)`.

This can be used to convert float samples to integers, integers
to log space, etc.

The core method to do so is to be able to cast
[`to_unit()`][neps.space.domain.Domain.to_unit] which takes
values to a unit interval [0, 1], and then to be able to cast values in [0, 1]
to the new domain with [`from_unit()`][neps.space.domain.Domain.from_unit].

There are some shortcuts implemented in `cast`, such as skipping going through
the unit interval if the domains are the same, as no transformation is needed.

The primary methods for creating a domain are

* [`Domain.floating(l, u, ...)`][neps.space.domain.Domain.floating] -
    Used for modelling float ranges
* [`Domain.integer(l, u, ...)`][neps.space.domain.Domain.integer] -
    Used for modelling integer ranges
* [`Domain.indices(n)`][neps.space.domain.Domain.indices] -
    Primarly used to model categorical choices

If you have a tensor of values, where each column corresponds to a different domain,
you can take a look at [`Domain.translate()`][neps.space.domain.Domain.translate]

If you need a unit-interval domain, please use the
[`Domain.unit_float()`][neps.space.domain.Domain.unit_float].
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from neps.space.encoding import ConfigEncoder

Number = int | float
V = TypeVar("V", int, float)


@dataclass(frozen=True)
class Domain(Generic[V]):
    """A domain for a value.

    The primary methods for creating a domain are

    * [`Domain.floating(l, u, ...)`][neps.space.domain.Domain.floating] -
        Used for modelling float ranges
    * [`Domain.integer(l, u, ...)`][neps.space.domain.Domain.integer] -
        Used for modelling integer ranges
    * [`Domain.indices(n)`][neps.space.domain.Domain.indices] -
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

    is_categorical: bool = False
    """Whether the domain is representing a categorical.

    The domain does not use this information directly, but it can be useful for external
    classes that consume Domain objects. This can only be set to `True` if the
    `cardinality` of the domain is finite, i.e. `bins` is not `None` OR `round`
    is `True` or the boundaries are both integers.
    """

    is_unit_float: bool = field(init=False, repr=False)
    is_int: bool = field(init=False, repr=False)
    length: V = field(init=False, repr=False)
    cardinality: int | None = field(init=False, repr=False)
    bounds: tuple[V, V] = field(init=False, repr=False)
    preffered_dtype: torch.dtype = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.lower, type(self.upper))
        is_int = isinstance(self.lower, int)
        object.__setattr__(self, "is_int", is_int)
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
            if self.is_categorical:
                raise ValueError(
                    "Categorical domain must have finite cardinality but"
                    " `bins` is `None` and `round` is `False` and"
                    " boundaries are not integers."
                )
        object.__setattr__(self, "cardinality", cardinality)

        preferred_dtype = torch.int64 if is_int else torch.float64
        object.__setattr__(self, "preffered_dtype", preferred_dtype)

        mid = self.from_unit(torch.tensor(0.5)).item()
        if is_int:
            mid = int(round(mid))

        object.__setattr__(self, "bounds", (self.lower, self.upper))

    @classmethod
    def floating(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
        is_categorical: bool = False,
    ) -> Domain[float]:
        """Create a domain for a range of float values.

        Args:
            lower: The lower bound of the domain.
            upper: The upper bound of the domain.
            log: Whether the domain is in log space.
            bins: The number of discrete bins to split the domain into.
            is_categorical: Whether the domain is representing a categorical.

        Returns:
            A domain for a range of float values.
        """
        return Domain(
            lower=float(lower),
            upper=float(upper),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            bins=bins,
            round=False,
            is_categorical=is_categorical,
        )

    @classmethod
    def integer(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
        is_categorical: bool = False,
    ) -> Domain[int]:
        """Create a domain for a range of integer values.

        Args:
            lower: The lower bound of the domain.
            upper: The upper bound of the domain (inclusive).
            log: Whether the domain is in log space.
            bins: The number of discrete bins to split the domain into.
            is_categorical: Whether the domain is representing a categorical.

        Returns:
            A domain for a range of integer values.
        """
        return Domain(
            lower=int(round(lower)),
            upper=int(round(upper)),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            round=True,
            bins=bins,
            is_categorical=is_categorical,
        )

    @classmethod
    def indices(cls, n: int, *, is_categorical: bool = False) -> Domain[int]:
        """Create a domain for a range of indices.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Args:
            n: The number of indices.
            is_categorical: Whether the domain is representing a categorical.

        Returns:
            A domain for a range of indices.
        """
        return Domain.integer(0, n - 1, is_categorical=is_categorical)

    def to_unit(self, x: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        """Transform a tensor of values from this domain to the unit interval [0, 1].

        Args:
            x: Tensor of values in this domain to convert.
            dtype: The dtype to convert to

        Returns:
            Same shape tensor with the values normalized to the unit interval [0, 1].
        """
        if dtype is None:
            dtype = torch.float64
        elif not dtype.is_floating_point:
            raise ValueError(f"Unit interval only allows floating dtypes, got {dtype}.")

        q = self.cardinality
        if self.is_unit_float and q is None:
            return x.to(dtype)

        if self.log_bounds is not None:
            x = torch.log(x)
            lower, upper = self.log_bounds
        else:
            lower, upper = self.lower, self.upper

        x = (x - lower) / (upper - lower)

        if q is not None:
            quantization_levels = torch.floor(x * q).clip(0, q - 1)
            x = quantization_levels / (q - 1)

        return x.type(dtype)

    def from_unit(self, x: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        """Transform a tensor of values from the unit interval [0, 1] to this domain.

        Args:
            x: A tensor of values in the unit interval [0, 1] to convert.
            dtype: The dtype to convert to

        Returns:
            Same shape tensor with the lifted into this domain.
        """
        dtype = dtype or self.preffered_dtype
        if self.is_unit_float:
            return x.to(dtype)

        q = self.cardinality
        if q is not None:
            quantization_levels = torch.floor(x * q).clip(0, q - 1)
            x = quantization_levels / (q - 1)

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

        if (x > upper).any():
            import warnings

            warnings.warn(  # noqa: B028
                "Decoded value is above the upper bound of the domain. "
                "Clipping to the upper bound. "
                "This is likely due floating point precision in `torch.exp(x)` "
                "with torch.float64."
            )
            x = torch.clip(x, max=self.upper)

        return x.type(dtype)

    def cast(self, x: Tensor, frm: Domain, *, dtype: torch.dtype | None = None) -> Tensor:
        """Cast a tensor of values frm the domain `frm` to this domain.

        If you need to cast a tensor of mixed domains, use
        [`Domain.translate()`][neps.space.domain.Domain.translate].

        Args:
            x: Tensor of values in the `frm` domain to cast to this domain.
            frm: The domain to cast from.
            dtype: The dtype to convert to

        Returns:
            Same shape tensor with the values cast to this domain.
        """
        dtype = dtype or self.preffered_dtype
        # NOTE: In general, we should always be able to go through the unit interval
        # [0, 1] to be able to transform between domains. However sometimes we can
        # bypass some steps, dependant on the domains, hence the ugliness...

        # Shortcut 1. (Same Domain)
        # We can shortcut out going through normalized space if all the boundaries and
        # they live on the same scale. However, if their bins don't line up, we will
        # have to go through unit space to figure out the bins
        same_bounds = self.lower == frm.lower and self.upper == frm.upper
        same_log_bounds = self.log_bounds == frm.log_bounds
        same_cardinality = self.cardinality == frm.cardinality
        if same_bounds and same_log_bounds and same_cardinality:
            if self.round:
                x = torch.round(x)
            return x.type(dtype)

        # Shortcut 2. (From normalized)
        # The domain we are coming from is already normalized, we only need to lift
        if frm.is_unit_float:
            return self.from_unit(x, dtype=dtype)  # type: ignore

        # Shortcut 3. (Log lift)
        # We can also shortcut out if the only diffrence is that we are coming frm the
        # log bounds of this domain. We dont care if where we came from was binned or not,
        # we just lift it up with `np.exp` and round if needed
        if (self.lower, self.upper) == frm.log_bounds and self.cardinality is None:
            x = torch.exp(x)
            if self.round:
                x = torch.round(x)
            return x.type(dtype)

        # Otherwise, through the unit interval we go
        lift = self.from_unit(frm.to_unit(x), dtype=dtype)
        return lift  # noqa: RET504

    @classmethod
    def unit_float(cls) -> Domain[float]:
        """Get a domain for the unit interval [0, 1]."""
        return UNIT_FLOAT_DOMAIN

    @classmethod
    def translate(
        cls,
        x: Tensor,
        frm: Domain | Iterable[Domain] | ConfigEncoder,
        to: Domain | Iterable[Domain] | ConfigEncoder,
        *,
        dtype: torch.dtype | None = None,
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
            dtype: The dtype of the converted tensor

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
            return to.cast(x, frm=frm, dtype=dtype)

        from neps.space.encoding import ConfigEncoder

        frm = (
            [frm] * ndims
            if isinstance(frm, Domain)
            else (frm.domains if isinstance(frm, ConfigEncoder) else list(frm))
        )
        to = (
            [to] * ndims
            if isinstance(to, Domain)
            else (to.domains if isinstance(to, ConfigEncoder) else list(to))
        )

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

        out = torch.empty_like(x, dtype=dtype)
        for i, (f, t) in enumerate(zip(frm, to, strict=False)):
            out[..., i] = t.cast(x[..., i], frm=f, dtype=dtype)

        return out

    def cast_one(self, x: float | int, frm: Domain) -> V:
        """Cast a single value from the domain `frm` to this domain.

        Args:
            x: Value in the `frm` domain to cast to this domain.
            frm: The domain to cast from.

        Returns:
            Value cast to this domain.
        """
        return self.cast(torch.tensor(x), frm=frm).item()  # type: ignore

    def to_unit_one(self, x: float | int) -> float:
        """Transform a single value from this domain to the unit interval [0, 1].

        Args:
            x: Value in this domain to convert.

        Returns:
            Value normalized to the unit interval [0, 1].
        """
        return self.to_unit(torch.tensor(x)).item()

    def as_integer_domain(self) -> Domain:
        """Get the integer version of this domain.

        !!! warning

            This is only possible if this domain has a finite cardinality
        """
        if self.cardinality is None:
            raise ValueError(
                "Cannot get integer representation of this domain as its"
                " cardinality is non-finite."
            )

        return Domain.indices(self.cardinality, is_categorical=self.is_categorical)


UNIT_FLOAT_DOMAIN = Domain.floating(0.0, 1.0)
