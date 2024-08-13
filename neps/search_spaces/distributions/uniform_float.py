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
class UniformFloatDistribution(Distribution[f64]):
    domain: Domain[f64]
    _pdf: f64 = field(repr=False)

    @override
    def raw_samples(self, n: int, seed: np.random.Generator) -> Arr[f64]:
        return seed.uniform(self.domain.lower, self.domain.upper, size=n)

    @override
    def likelihood(self, value: Arr[f64]) -> Arr[f64]:
        return np.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            F64_ZERO,
        )

    @classmethod
    def new(cls, lower: Number, upper: Number) -> UniformFloatDistribution:
        return cls(Domain.float(lower, upper), _pdf=f64(1.0 / (upper - lower)))


UNIT_UNIFORM_FLOAT = UniformFloatDistribution.new(0.0, 1.0)
