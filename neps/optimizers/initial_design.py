"""Initial design of points for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

if TYPE_CHECKING:
    import torch

    from neps.priors import Prior
    from neps.search_spaces.domain import Domain


@dataclass
class PriorInitialDesign(InitialDesign):
    """Sample from a prior distribution."""

    prior: Prior
    """The prior to sample from."""

    # TODO: Right now we don't have a way to set the seed temporarily
    seed: int | None = None
    """The seed for sampling."""

    @override
    def sample(self, n: int) -> torch.Tensor:
        return self.prior.sample(n)

    @property
    @override
    def sample_domain(self) -> list[Domain]:
        return self.prior.domains
