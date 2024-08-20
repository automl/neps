from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Sequence
from typing_extensions import override

import torch
from torch import Tensor

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import Domain

if TYPE_CHECKING:
    from neps.utils.types import Number


@dataclass(frozen=True)
class WeightedIntsDistribution(Distribution[int]):
    # NOTE: Having a Million weights is very resource intense and super slow
    # for sampling, especially given our common use case is to have only one weight
    # with the rest being uniform. 100 is well out of scope for what was intended,
    # as this is mostly intended for categoricals.
    # If we need this, then we should make a more efficient implementation,
    # such as one that uniform samples and then with probability `weight`
    # replaces the value with the favoured value.
    LIMIT_FOR_WEIGHTED_INTS: ClassVar[int] = 200

    domain: Domain[int]
    weights: Tensor

    @override
    def sample(self, n: int, to: Domain, *, seed: torch.Generator) -> Tensor:
        rand_tensor = torch.multinomial(
            self.weights,
            n,
            replacement=True,
            generator=seed,
        )
        return to.cast(rand_tensor, frm=self.domain)

    @override
    def likelihood(self, value: Tensor) -> Tensor:
        valid_indices = torch.logical_and(
            value >= self.domain.lower, value <= self.domain.upper
        )
        psuedo_indices = torch.where(valid_indices, value, 0)
        probs = self.weights[psuedo_indices]
        return torch.where(valid_indices, probs, 0)

    @classmethod
    def new(cls, weights: Sequence[Number] | Tensor) -> WeightedIntsDistribution:
        if len(weights) > cls.LIMIT_FOR_WEIGHTED_INTS:
            raise ValueError(
                f"Having {len(weights)} weights is very resource intense and slow"
                " for sampling. Consider using a more efficient implementation"
                " if you need this many weights.",
            )
        return cls(
            weights=torch.as_tensor(weights, dtype=torch.float64),
            domain=Domain.indices(len(weights)),
        )

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
        dist = torch.full(size=(n,), fill_value=rest, dtype=torch.float64)
        dist[favoured] = confidence
        return cls(weights=dist, domain=Domain.indices(n))
