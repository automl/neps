from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np

from neps.search_spaces.domain import Domain
from neps.utils.types import Number, f64, i64

V = TypeVar("V", i64, f64)
F64_ZERO = f64(0.0)


if TYPE_CHECKING:
    from neps.utils.types import Array


@dataclass(frozen=True)
class Distribution(ABC, Generic[V]):
    domain: Domain[V]

    @abstractmethod
    def sample(self, n: int, seed: np.random.Generator) -> Array[V]: ...

    @abstractmethod
    def likelihood(self, value: Array[V]) -> Array[f64]: ...


@dataclass(frozen=True)
class UniformFloatDistribution(Distribution[f64]):
    domain: Domain[f64]
    _pdf: f64 = field(repr=False)

    def sample(self, n: int, seed: np.random.Generator) -> Array[f64]:
        return seed.uniform(self.domain.lower, self.domain.upper, size=n)

    def likelihood(self, value: Array[f64]) -> Array[f64]:
        return np.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            F64_ZERO,
        )

    @classmethod
    def new(cls, lower: Number, upper: Number) -> UniformFloatDistribution:
        return cls(Domain.float(lower, upper), _pdf=f64(1.0 / (upper - lower)))


@dataclass(frozen=True)
class UniformIntDistribution(Distribution[i64]):
    domain: Domain[i64]
    _pdf: f64 = field(repr=False)

    def sample(self, n: int, seed: np.random.Generator) -> Array[i64]:
        return seed.integers(self.domain.lower, self.domain.upper, size=n)

    def likelihood(self, value: Array[i64]) -> Array[f64]:
        return np.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            F64_ZERO,
        )

    @classmethod
    def new(cls, lower: Number, upper: Number) -> UniformIntDistribution:
        return cls(Domain.int(lower, upper), _pdf=f64(1.0 / (upper - lower)))


@dataclass(frozen=True)
class TruncNormDistribution(Distribution[f64]):
    domain: Domain[f64]
    truncnorm: Any
    mean: f64
    std: f64

    def sample(self, n: int, seed: np.random.Generator) -> Array[f64]:
        return self.truncnorm.rvs(size=n, random_state=seed)

    def likelihood(self, value: Array[f64]) -> Array[f64]:
        return self.truncnorm.pdf(value)

    @classmethod
    def new(
        cls,
        mean: Number,
        std: Number,
        lower: Number,
        upper: Number,
    ) -> TruncNormDistribution:
        from scipy.stats import truncnorm

        return cls(
            domain=Domain.float(lower, upper),
            truncnorm=truncnorm(
                a=(lower - mean) / std,
                b=(upper - mean) / std,
                loc=mean,
                scale=std,
            ),
            mean=f64(mean),
            std=f64(std),
        )


@dataclass(frozen=True)
class WeightedIntsDistribution(Distribution[i64]):
    domain: Domain[i64]
    probabilities: Array[f64]

    def sample(self, n: int, seed: np.random.Generator) -> Array[i64]:
        return seed.choice(
            a=self.domain.upper,
            replace=False,
            size=n,
            p=self.probabilities,
        )

    def likelihood(self, value: Array[i64]) -> Array[f64]:
        valid_indices = np.logical_and(
            value >= self.domain.lower, value <= self.domain.upper
        )
        psuedo_indices = np.where(valid_indices, value, 0)
        probs = self.probabilities[psuedo_indices]
        return np.where(valid_indices, probs, F64_ZERO)

    @classmethod
    def new(cls, weights: Sequence[Number] | Array[f64]) -> WeightedIntsDistribution:
        weights = np.asarray(weights)
        probabilities = weights / np.sum(weights)
        return cls(probabilities=probabilities, domain=Domain.int(0, len(weights)))


UNIT_UNIFORM = UniformFloatDistribution.new(0.0, 1.0)
