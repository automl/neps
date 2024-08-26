# TODO: Could theoretically implement dtype,device,out for all methods here but
# would need to be careful not to accidentally send to and from GPU.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar
from typing_extensions import TypeAlias

import torch
from torch import Tensor

if TYPE_CHECKING:
    from neps.search_spaces.distributions.truncnorm import TruncNormDistribution
    from neps.search_spaces.distributions.uniform_float import (
        UniformFloatDistribution,
    )
    from neps.search_spaces.distributions.uniform_int import UniformIntDistribution
    from neps.search_spaces.distributions.weighted_ints import WeightedIntsDistribution


Number = int | float
V = TypeVar("V", int, float)
V2 = TypeVar("V2", int, float)


@dataclass(frozen=True)
class NumberDomain(Generic[V]):
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
    ) -> NumberDomain[float]:
        return NumberDomain(
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
    ) -> NumberDomain[int]:
        return NumberDomain(
            lower=int(round(lower)),
            upper=int(round(upper)),
            log_bounds=(math.log(lower), math.log(upper)) if log else None,
            round=True,
            bins=bins,
        )

    @classmethod
    def indices(cls, n: int) -> NumberDomain[int]:
        """Create a domain for a range of indices.

        Like range based functions this domain is inclusive of the lower bound
        and exclusive of the upper bound.

        Use this method to create a domain for indices
        """
        return NumberDomain.int(0, n - 1)

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
        if self.dtype == torch.int64 and self.bins is None:
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
            _std = float(1 - confidence)  # type: ignore
            _is_normalized = True
        else:
            _std = float(std)
            _is_normalized = False

        # (Log Lift) - sample on it's log domain
        if self.log_bounds is not None:
            return TruncNormDistribution.new(
                lower=self.log_bounds[0],
                center=math.log(center),
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
            center=math.log(center),
            upper=self.upper,
            std=_std,
            std_is_normalized=_is_normalized,
        )
        return truncnorm.normalize()

    def weighted_indices_distribution(
        self, center_index: int, *, confidence: float
    ) -> WeightedIntsDistribution:
        from neps.search_spaces.distributions import WeightedIntsDistribution

        if self.cardinality is None:
            raise ValueError(
                "Cannot create a weighted distribution for a continuous domain!"
            )
        if not isinstance(center_index, int):
            raise ValueError(
                f"Center index must be an integer of type {self.dtype} to"
                " create a weighted distribution!"
            )
        assert 0 <= confidence <= 1

        return WeightedIntsDistribution.with_favoured(
            n=self.cardinality,
            favoured=int(round(center_index)),
            confidence=confidence,
        )

    @classmethod
    def unit_float(cls) -> NumberDomain[float]:
        return UNIT_FLOAT_DOMAIN


UNIT_FLOAT_DOMAIN = NumberDomain.float(0.0, 1.0)

Domain: TypeAlias = NumberDomain
