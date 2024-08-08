from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np

from neps.utils.types import f64, i64

if TYPE_CHECKING:
    from neps.utils.types import Array, Number

V = TypeVar("V", f64, i64)
V2 = TypeVar("V2", f64, i64)


@dataclass(frozen=True)
class Domain(Generic[V]):
    lower: V
    upper: V
    log_bounds: tuple[f64, f64] | None = None
    bins: int | None = None

    dtype: type[V] = field(init=False, repr=False)
    is_int: bool = field(init=False, repr=False)
    is_unit: bool = field(init=False, repr=False)
    is_log: bool = field(init=False, repr=False)
    cardinality: int | None = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "is_int", self.lower.dtype == i64)
        object.__setattr__(self, "is_unit", self.lower == 0 and self.upper == 1)
        object.__setattr__(self, "is_log", self.log_bounds is not None)
        object.__setattr__(self, "dtype", i64 if self.is_int else f64)

        if self.bins:
            cardinality = self.bins
        elif self.is_int:
            cardinality = int(self.upper - self.lower + 1)
        else:
            cardinality = None

        object.__setattr__(self, "cardinality", cardinality)

    @classmethod
    def float(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
    ) -> Domain[f64]:
        return Domain(
            lower=f64(lower),
            upper=f64(upper),
            log_bounds=(np.log(lower), np.log(upper)) if log else None,
            bins=bins,
        )

    @classmethod
    def range(cls, lower: int | i64, upper: int | i64) -> Domain[i64]:
        """Create a domain for a range of integers.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Use this method to create a domain for ranges
        """
        return Domain(lower=i64(lower), upper=i64(upper))

    @classmethod
    def int(
        cls,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        bins: int | None = None,
    ) -> Domain[i64]:
        return Domain(
            lower=i64(np.rint(lower)),
            upper=i64(np.rint(upper)),
            log_bounds=(np.log(lower), np.log(upper)) if log else None,
            bins=bins,
        )

    @overload
    def normalize(self, x: Array[V]) -> Array[f64]: ...
    @overload
    def normalize(self, x: V) -> f64: ...

    def normalize(self, x: V | Array[V]) -> f64 | Array[f64]:
        if self.is_unit:
            return x  # type: ignore

        if self.log_bounds is not None:
            x = np.log(x)
            lower, upper = self.log_bounds
        else:
            lower, upper = self.lower, self.upper

        return (x - lower) / (upper - lower)

    @overload
    def from_normalized(self, x: Array[f64]) -> Array[V]: ...
    @overload
    def from_normalized(self, x: f64) -> V: ...

    def from_normalized(self, x: f64 | Array[f64]) -> V | Array[V]:
        if self.is_unit:
            return x  # type: ignore

        bins = self.bins
        if bins is not None:
            quantization_levels = np.floor(x * bins).clip(0, bins - 1)
            x = quantization_levels / (bins - 1)

        # Now we scale to the new domain
        if self.log_bounds is not None:
            lower, upper = self.log_bounds
            x = x * (upper - lower) + lower
            x = np.exp(x)
        else:
            lower, upper = self.lower, self.upper
            x = x * (upper - lower) + lower

        if self.is_int:
            x = np.rint(x)

        return x.astype(self.dtype)

    @overload
    def cast(self, x: Array[V2], frm: Domain[V2]) -> Array[V]: ...
    @overload
    def cast(self, x: V2, frm: Domain[V2]) -> V: ...

    def cast(self, x: V2 | Array[V2], frm: Domain[V2]) -> V | Array[V]:
        return self.from_normalized(frm.normalize(x))


UNIT_FLOAT = Domain.float(0, 1)
