from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from neps.utils.types import f64, i64

V = TypeVar("V", i64, f64)
V2 = TypeVar("V2", i64, f64)


if TYPE_CHECKING:
    import numpy as np

    from neps.search_spaces.domain import Domain
    from neps.utils.types import Arr


@dataclass(frozen=True)
class Distribution(ABC, Generic[V]):
    domain: Domain[V]

    def sample(self, n: int, *, to: Domain[V2], seed: np.random.Generator) -> Arr[V2]:
        return to.cast(self.raw_samples(n, seed), frm=self.domain)

    @abstractmethod
    def raw_samples(self, n: int, seed: np.random.Generator) -> Arr[V]: ...

    @abstractmethod
    def likelihood(self, value: Arr[V]) -> Arr[f64]: ...
