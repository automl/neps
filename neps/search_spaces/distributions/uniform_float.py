from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import override

import torch
from torch import Tensor

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain

INT_HIGH = 1_000_000


@dataclass(frozen=True)
class UniformFloatDistribution(Distribution[float]):
    domain: Domain[float]
    _pdf: float = field(repr=False)

    @override
    def sample(self, n: int, to: Domain, seed: torch.Generator) -> Tensor:
        # This creates samples in a unit float domain, rather than
        # the `.domain` attribute of this distribution. Rather than scale
        # up twice, we just scale directly form the UNIT_FLOAT_DOMAIN
        # We still however need the `.domain` attribute for `likelihood`
        unit_samples = torch.rand(n, generator=seed)
        return to.cast(unit_samples, UNIT_FLOAT_DOMAIN)

    @override
    def likelihood(self, value: Tensor) -> Tensor:
        return torch.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            0.0,
        )

    @classmethod
    def new(cls, lower: int | float, upper: int | float) -> UniformFloatDistribution:
        _pdf = 1.0 / (upper - lower)
        return cls(Domain.float(lower, upper), _pdf=_pdf)

    @classmethod
    def unit_distribution(cls) -> UniformFloatDistribution:
        return UNIT_UNIFORM_FLOAT


UNIT_UNIFORM_FLOAT = UniformFloatDistribution.new(0.0, 1.0)
