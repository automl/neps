from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
from typing_extensions import Protocol

V = TypeVar("V", int, float)


if TYPE_CHECKING:
    from torch import Generator, Tensor

    from neps.search_spaces.domain import Domain


class Distribution(Protocol[V]):
    @property
    def domain(self) -> Domain[V]: ...

    def sample(self, n: int, to: Domain, *, seed: Generator) -> Tensor: ...

    def likelihood(self, value: Tensor) -> Tensor: ...
