from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np

from neps.utils.types import f64, i64

if TYPE_CHECKING:
    from neps.search_spaces.distributions.truncnorm import TruncNormDistribution
    from neps.search_spaces.distributions.uniform_float import (
        UniformFloatDistribution,
    )
    from neps.search_spaces.distributions.uniform_int import UniformIntDistribution
    from neps.search_spaces.distributions.weighted_ints import WeightedIntsDistribution
    from neps.utils.types import Arr, Number


V = TypeVar("V", f64, i64)
V2 = TypeVar("V2", f64, i64)


@dataclass(frozen=True)
class Domain(Generic[V]):
    lower: V
    upper: V
    round: bool
    log_bounds: tuple[f64, f64] | None = None
    bins: int | None = None

    dtype: type[V] = field(init=False, repr=False)
    is_unit: bool = field(init=False, repr=False)
    midpoint: V = field(init=False, repr=False)
    is_log: bool = field(init=False, repr=False)
    length: V = field(init=False, repr=False)
    cardinality: int | None = field(init=False, repr=False)

    def __post_init__(self):
        assert self.lower.dtype == self.upper.dtype
        object.__setattr__(self, "is_unit", self.lower == 0 and self.upper == 1)
        object.__setattr__(self, "is_log", self.log_bounds is not None)
        object.__setattr__(self, "dtype", i64 if self.lower.dtype == i64 else f64)
        object.__setattr__(self, "length", self.upper - self.lower)

        if self.bins:
            cardinality = self.bins
        elif self.round:
            cardinality = int(self.upper - self.lower + 1)
        else:
            cardinality = None

        object.__setattr__(self, "cardinality", cardinality)
        object.__setattr__(self, "midpoint", self.from_unit(f64(0.5)))

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
    ) -> Domain[i64]:
        return Domain(
            lower=i64(np.rint(lower)),
            upper=i64(np.rint(upper)),
            log_bounds=(np.log(lower), np.log(upper)) if log else None,
            round=True,
            bins=bins,
        )

    @classmethod
    def indices(cls, n: int) -> Domain[i64]:
        """Create a domain for a range of indices.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Use this method to create a domain for indices
        """
        return Domain.int(0, n - 1)

    def closest(self, x: V | Arr) -> V | Arr[V]:
        if self.bins is None:
            return np.clip(x, self.lower, self.upper)

        unit = self.to_unit(x)
        return self.from_unit(unit)

    @overload
    def to_unit(self, x: Arr[V]) -> Arr[f64]: ...
    @overload
    def to_unit(self, x: V) -> f64: ...

    def to_unit(self, x: V | Arr[V]) -> f64 | Arr[f64]:
        if self.is_unit:
            return x  # type: ignore

        if self.log_bounds is not None:
            x = np.log(x)
            lower, upper = self.log_bounds
        else:
            lower, upper = self.lower, self.upper

        return (x - lower) / (upper - lower)

    @overload
    def from_unit(self, x: Arr[f64]) -> Arr[V]: ...
    @overload
    def from_unit(self, x: f64) -> V: ...

    def from_unit(self, x: f64 | Arr[f64]) -> V | Arr[V]:
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

        if self.round:
            x = np.rint(x)

        return x.astype(self.dtype)

    @overload
    def cast(self, x: Arr[V2], frm: Domain[V2]) -> Arr[V]: ...
    @overload
    def cast(self, x: V2, frm: Domain[V2]) -> V: ...

    def cast(self, x: V2 | Arr[V2], frm: Domain[V2]) -> V | Arr[V]:
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
                x = np.rint(x)
            return x.astype(self.dtype)

        # Shortcut 2. (From normalized)
        # The domain we are coming from is already normalized, we only need to lift
        if frm.is_unit:
            return self.from_unit(x)  # type: ignore

        # Shortcut 3. (Log lift)
        # We can also shortcut out if the only diffrence is that we are coming frm the
        # log bounds of this domain. We dont care if where we came from was binned or not,
        # we just lift it up with `np.exp` and round if needed
        if (self.lower, self.upper) == frm.log_bounds and self.bins is None:
            x = np.exp(x)
            if self.round:
                x = np.rint(x)
            return x.astype(self.dtype)

        # Otherwise, through the unit interval we go
        norm = frm.to_unit(x)
        lift = self.from_unit(norm)
        return lift  # noqa: RET504

    def uniform_distribution(self) -> UniformFloatDistribution | UniformIntDistribution:
        from neps.search_spaces.distributions import (
            UNIT_UNIFORM,
            UniformFloatDistribution,
            UniformIntDistribution,
        )

        # (Log Lift) - sample on it's log domain
        if self.log_bounds is not None:
            return UniformFloatDistribution.new(*self.log_bounds)

        # (Same Domain) - Just sample integers
        if self.dtype == i64 and self.bins is None:
            return UniformIntDistribution.new(self.lower, self.upper)

        # NOTE: There's a possibility where you could use an integer distribution for
        # binned domains, however the cost of sampling integers and casting is likely
        # higher than just casting from normalized domain. Would need to verify this
        # In any case, Normalized Uniform Float is a safe choice

        # (From Normalized)
        return UNIT_UNIFORM

    def unit_uniform_distribution(self) -> UniformFloatDistribution:
        from neps.search_spaces.distributions import UNIT_UNIFORM

        return UNIT_UNIFORM

    def truncnorm_distribution(
        self,
        center: Number,
        *,
        confidence: float | None = None,
        std: float | None = None,
    ) -> TruncNormDistribution:
        from neps.search_spaces.distributions import TruncNormDistribution

        # If you need a unit one, create this and then call `normalize()` on it.
        if std is None and confidence is None:
            raise ValueError(
                "Must specify either `std` in (lower, upper) or `confidence` in (0, 1)"
            )

        if std is None:
            assert 0 <= confidence <= 1  # type: ignore
            _std = f64(1 - confidence)  # type: ignore
            _is_normalized = True
        else:
            _std = f64(std)  # type: ignore
            _is_normalized = False

        # (Log Lift) - sample on it's log domain
        if self.log_bounds is not None:
            return TruncNormDistribution.new(
                lower=self.log_bounds[0],
                center=np.log(center),
                upper=self.log_bounds[1],
                std=_std,
                std_is_normalized=_is_normalized,
            )

        # NOTE: There's a possibility where you could use an integer distribution for
        # binned domains, however the cost of sampling integers and casting is likely
        # higher than just casting from normalized domain. Would need to verify this
        # In any case, Normalized Uniform Float is a safe choice

        # (From Normalized)
        truncnorm = TruncNormDistribution.new(
            lower=self.lower,
            center=f64(center),
            upper=self.upper,
            std=_std,
            std_is_normalized=_is_normalized,
        )
        return truncnorm.normalize()

    def weighted_indices_distribution(
        self,
        center_index: int | i64,
        *,
        confidence: float,
    ) -> WeightedIntsDistribution:
        from neps.search_spaces.distributions import WeightedIntsDistribution

        if self.cardinality is None:
            raise ValueError(
                "Cannot create a weighted distribution for a continuous domain!"
            )
        if not isinstance(center_index, int) or (
            hasattr(self, "dtype") and not isinstance(center_index, self.dtype)
        ):
            raise ValueError(
                f"Center index must be an integer of type {self.dtype} to"
                " create a weighted distribution!"
            )
        assert 0 <= confidence <= 1

        return WeightedIntsDistribution.with_favoured(
            n=self.cardinality,
            favoured=int(np.rint(center_index)),
            confidence=confidence,
        )

    @classmethod
    def unit_float(cls) -> Domain[f64]:
        return UNIT_FLOAT_DOMAIN


UNIT_FLOAT_DOMAIN = Domain.float(0.0, 1.0)
