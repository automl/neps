from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Sequence, TypeVar
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
class WeightedIntsDistribution(Distribution[i64]):
    # NOTE: Having a Million weights is very resource intense and super slow
    # for sampling, especially given our common use case is to have only one weight
    # with the rest being uniform. 100 is well out of scope for what was intended,
    # as this is mostly intended for categoricals.
    # If we need this, then we should make a more efficient implementation,
    # such as one that uniform samples and then with probability `weight`
    # replaces the value with the favoured value.
    LIMIT_FOR_WEIGHTED_INTS: ClassVar[int] = 200

    domain: Domain[i64]
    probabilities: Arr[f64]

    @override
    def raw_samples(self, n: int, seed: np.random.Generator) -> Arr[i64]:
        _range = self.domain.upper - self.domain.lower + 1
        return seed.choice(
            a=_range,
            replace=False,
            size=n,
            p=self.probabilities,
        )

    @override
    def likelihood(self, value: Arr[i64]) -> Arr[f64]:
        valid_indices = np.logical_and(
            value >= self.domain.lower, value <= self.domain.upper
        )
        psuedo_indices = np.where(valid_indices, value, 0)
        probs = self.probabilities[psuedo_indices]
        return np.where(valid_indices, probs, F64_ZERO)

    @classmethod
    def new(cls, weights: Sequence[Number] | Arr[f64]) -> WeightedIntsDistribution:
        if len(weights) > cls.LIMIT_FOR_WEIGHTED_INTS:
            raise ValueError(
                f"Having {len(weights)} weights is very resource intense and slow"
                " for sampling. Consider using a more efficient implementation"
                " if you need this many weights.",
            )
        weights = np.asarray(weights)
        probabilities = weights / np.sum(weights)
        return cls(probabilities=probabilities, domain=Domain.indices(len(weights)))

    @classmethod
    def with_favoured(
        cls,
        n: int,
        favoured: int,
        confidence: float,
    ) -> WeightedIntsDistribution:
        if n > cls.LIMIT_FOR_WEIGHTED_INTS:
            raise ValueError(
                f"Having {n} weights is very resource intense and slow"
                " for sampling. Consider using a more efficient implementation"
                " if you need this many weights.",
            )

        assert 0.0 <= confidence <= 1.0
        remaining = 1.0 - confidence
        rest = remaining / (n - 1)
        if confidence < rest:
            warnings.warn(
                f"Weight {confidence} is less than the rest {rest}."
                " This will make the favoured value less likely to be sampled"
                " than the rest of the values.",
                UserWarning,
                stacklevel=2,
            )
        dist = np.full(n, rest)
        dist[favoured] = confidence
        return cls(probabilities=dist, domain=Domain.indices(n))
