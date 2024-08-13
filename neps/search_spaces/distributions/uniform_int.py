from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import override

import numpy as np

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import Domain
from neps.utils.types import Number, f64, i64

V = TypeVar("V", i64, f64)
F64_ZERO = f64(0.0)


if TYPE_CHECKING:
    from neps.utils.types import Arr


@dataclass(frozen=True)
class UniformIntDistribution(Distribution[i64]):
    domain: Domain[i64]
    _pdf: f64 = field(repr=False)

    @override
    def raw_samples(self, n: int, seed: np.random.Generator) -> Arr[i64]:
        return seed.integers(self.domain.lower, self.domain.upper, size=n)

    @override
    def likelihood(self, value: Arr[i64]) -> Arr[f64]:
        return np.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            F64_ZERO,
        )

    @classmethod
    def indices(cls, n: int) -> UniformIntDistribution:
        return cls(Domain.int(0, n - 1), _pdf=f64(1.0 / n))

    @classmethod
    def new(cls, lower: Number, upper: Number) -> UniformIntDistribution:
        return cls(Domain.int(lower, upper), _pdf=f64(1.0 / (upper - lower)))
