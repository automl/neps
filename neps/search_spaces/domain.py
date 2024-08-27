"""A class representing a domain, a range for a value + properties.

Some properties include:

* The lower and upper bounds of the domain.
* Whether the domain is a log domain.
* Whether the domain is float/int.
* The midpoint of the domain.
* Whether the domain is split into bins.

With that, the primary method of a domain is to be able to cast
values from one to domain to another,
e.g. `values_a = domain_a.cast(values_b, frm=domain_b)`.

This can be used to convert float samples to integers, integers
to log space, etc.

The core method to do so is to be able to cast `to_unit` which takes
values to a unit interval [0, 1], and then to be able to cast values in [0, 1]
to the new domain with `from_unit`.

There are some shortcuts implemented in `cast`, such as skipping going through
the unit interval if the domains are the same, as no transformation is needed.

The primary methods for creating a domain are

* `Domain.float(l, u, ...)` - Used for modelling float ranges
* `Domain.int(l, u, ...)` - Used for modelling integer ranges
* `Domain.indices(n)` - Primarly used to model categorical choices

If you have a tensor of values, where each column corresponds to a different domain,
you can take a look at `Domain.cast_many` to cast all the values in one go.

If you need a unit-interval domain, please use the `Domain.unit_float()` or
`UNIT_FLOAT_DOMAIN` constant.
"""

# TODO: Could theoretically implement dtype,device,out for all methods here but
# would need to be careful not to accidentally send to and from GPU.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, Sequence, TypeVar

import torch
from torch import Tensor

Number = int | float
V = TypeVar("V", int, float)
V2 = TypeVar("V2", int, float)


@dataclass(frozen=True)
class Domain(Generic[V]):
    lower: V
    upper: V
    round: bool
    log_bounds: tuple[float, float] | None = None
    bins: int | None = None

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
        return Domain(
            lower=int(round(lower)),
            upper=int(round(upper)),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            round=True,
            bins=bins,
        )

    @classmethod
    def indices(cls, n: int) -> Domain[int]:
        """Create a domain for a range of indices.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Use this method to create a domain for indices
        """
        return Domain.int(0, n - 1)

    def to_unit(self, x: Tensor) -> Tensor:
        if self.is_unit_float:
            return x  # type: ignore

        if self.log_bounds is not None:
            x = torch.log(x)
            lower, upper = self.log_bounds
        else:
            lower, upper = self.lower, self.upper

        return (x - lower) / (upper - lower)

    def from_unit(self, x: Tensor) -> Tensor:
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

    def cast(
        self,
        x: Tensor,
        frm: Domain,
    ) -> Tensor:
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
            return x.type(self.dtype)

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
        return UNIT_FLOAT_DOMAIN

    @classmethod
    def cast_many(
        cls, x: Tensor, frm: Domain | Sequence[Domain], to: Domain | Sequence[Domain]
    ) -> Tensor:
        """Cast a tensor of mixed domains to a new set of mixed domains.

        Args:
            x: Tensor of shape (n_samples, n_dims) with each dim `i` corresponding
                to the domain `frm[i]`.
            frm: List of domains to cast from. If list, must be length of `n_dims`,
                otherwise we assume the single domain provided is the one to be used
                across all dimensions.
            to: List of domains to cast to. If list, must be length as `n_dims`,
                otherwise we assume the single domain provided is the one to be used
                across all dimensions.

        Returns:
            Tensor of shape (n_samples, n_dims) with each dim `i` transformed
            from the domain `frm[i]` to the domain `to[i]`.
        """
        if x.ndim == 1:
            raise ValueError(
                "Expected a 2D tensor of shape (n_samples, n_dims), got a 1D tensor."
            )

        if isinstance(frm, Sequence) and len(frm) != x.shape[1]:
            raise ValueError(
                "The number of domains in `frm` must match the number of tensors"
                " if provided as a list."
                f" Expected {x.shape[1]}, got {len(frm)}."
            )

        if isinstance(to, Sequence) and len(to) != x.shape[1]:
            raise ValueError(
                "The number of domains in `to` must match the number of tensors"
                " if provided as a list."
                f" Expected {x.shape[1]}, got {len(to)}."
            )

        # If both are not a list, we can just cast the whole tensor
        if not isinstance(frm, Sequence) and not isinstance(to, Sequence):
            return to.cast(x, frm=frm)

        # Otherwise, we need to go column by column
        if isinstance(frm, Domain):
            frm = [frm] * x.shape[1]
        if isinstance(to, Domain):
            to = [to] * x.shape[1]

        buffer = torch.empty_like(x)
        for i, (f, t) in enumerate(zip(frm, to)):
            buffer[:, i] = t.cast(x[:, i], frm=f)

        return buffer


UNIT_FLOAT_DOMAIN = Domain.float(0.0, 1.0)
