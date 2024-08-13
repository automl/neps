from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import override

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import Domain
from neps.utils.types import Number, f64, i64

V = TypeVar("V", i64, f64)
F64_ZERO = f64(0.0)


if TYPE_CHECKING:
    import numpy as np

    from neps.utils.types import Arr


@lru_cache
def _truncnorm(a: f64, b: f64, loc: f64, scale: f64) -> Any:
    from scipy.stats import truncnorm

    return truncnorm(a=a, b=b, loc=loc, scale=scale)


@dataclass(frozen=True)
class TruncNormDistribution(Distribution[f64]):
    domain: Domain[f64]
    center: f64
    std: f64
    truncnorm: Any

    @override
    def raw_samples(self, n: int, seed: np.random.Generator) -> Arr[f64]:
        return self.truncnorm.rvs(size=n, random_state=seed)

    @override
    def likelihood(self, value: Arr[f64]) -> Arr[f64]:
        return self.truncnorm.pdf(value)

    def normalize(self) -> TruncNormDistribution:
        # Send to unit domain
        center = self.domain.from_unit(self.center)
        std = self.std / self.domain.length

        return TruncNormDistribution(
            domain=Domain.unit_float(),
            center=center,
            std=std,
            truncnorm=_truncnorm(
                a=(0 - center) / std,
                b=(1 - center) / std,
                loc=center,
                scale=std,
            ),
        )

    def with_center_and_confidence(
        self,
        center: Number,
        confidence: float,
    ) -> TruncNormDistribution:
        assert 0 <= confidence <= 1
        assert self.domain.lower <= center <= self.domain.upper
        std = f64(1 - confidence)
        center = f64(center)
        return TruncNormDistribution(
            domain=self.domain,
            center=center,
            std=std,
            truncnorm=_truncnorm(
                a=(self.domain.lower - center) / std,
                b=(self.domain.upper - center) / std,
                loc=center,
                scale=std,
            ),
        )

    @classmethod
    def new(
        cls,
        lower: Number,
        center: Number,
        upper: Number,
        *,
        std: Number,
        std_is_normalized: bool,
    ) -> TruncNormDistribution:
        assert lower <= center <= upper, f"{lower} <= {center} <= {upper}"
        center = f64(center)

        if std_is_normalized:
            assert 0 <= std <= 1
            std = f64((upper - lower) * std)
        else:
            assert std > 0
            std = f64(std)

        return cls(
            domain=Domain.float(lower, upper),
            center=center,
            std=std,
            truncnorm=_truncnorm(
                a=(lower - center) / std,
                b=(upper - center) / std,
                loc=center,
                scale=std,
            ),
        )
