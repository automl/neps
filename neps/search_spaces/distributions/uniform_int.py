from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import override

import torch
from torch import Tensor

from neps.search_spaces.distributions.distribution import Distribution
from neps.search_spaces.domain import Domain

if TYPE_CHECKING:
    from neps.utils.types import Number


@dataclass(frozen=True)
class UniformIntDistribution(Distribution[int]):
    domain: Domain[int]
    _pdf: float = field(repr=False)

    @override
    def sample(self, n: int, to: Domain, *, seed: torch.Generator) -> Tensor:
        samples = torch.randint(
            self.domain.lower,
            self.domain.upper,
            size=(n,),
            generator=seed,
        )
        return to.cast(samples, frm=self.domain)

    @override
    def likelihood(self, value: Tensor) -> Tensor:
        return torch.where(
            (value >= self.domain.lower) & (value <= self.domain.upper),
            self._pdf,
            0.0,
        )

    @classmethod
    def indices(cls, n: int) -> UniformIntDistribution:
        return cls(Domain.int(0, n - 1), _pdf=1.0 / n)

    @classmethod
    def new(cls, lower: Number, upper: Number) -> UniformIntDistribution:
        return cls(Domain.int(lower, upper), _pdf=1.0 / (upper - lower))
