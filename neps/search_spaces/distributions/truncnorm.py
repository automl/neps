from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch
from torch import Tensor

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import Domain

if TYPE_CHECKING:
    from neps.utils.types import Number

INT_HIGH = 1_000_000


@lru_cache
def _truncnorm(a: float, b: float, loc: float, scale: float) -> Any:
    from scipy.stats import truncnorm

    return truncnorm(a=a, b=b, loc=loc, scale=scale)


@dataclass(frozen=True)
class TruncNormDistribution(Distribution[float]):
    domain: Domain[float]
    center: float
    std: float
    truncnorm: Any

    @override
    def sample(self, n: int, seed: torch.Generator) -> Tensor:
        random_state = torch.randint(INT_HIGH, size=(1,), generator=seed)
        rv = self.truncnorm.rvs(size=n, random_state=random_state.item())
        return torch.tensor(rv, dtype=self.domain.dtype)

    @override
    def likelihood(self, value: Tensor) -> Tensor:
        return self.truncnorm.pdf(value.numpy())

    def normalize(self) -> TruncNormDistribution:
        # Send to unit domain
        center = float(self.domain.from_unit(torch.tensor(self.center)).item())
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
        std = 1 - confidence
        center = float(center)
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
        center = float(center)

        if std_is_normalized:
            assert 0 <= std <= 1
            std = float((upper - lower) * std)
        else:
            assert std > 0
            std = float(std)

        return cls(
            domain=Domain.float(float(lower), float(upper)),
            center=center,
            std=std,
            truncnorm=_truncnorm(
                a=(lower - center) / std,
                b=(upper - center) / std,
                loc=center,
                scale=std,
            ),
        )
