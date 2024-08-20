from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
from typing_extensions import Protocol

if TYPE_CHECKING:
    import numpy as np

    from neps.search_spaces.config import Config
    from neps.utils.types import Number


@dataclass
class Sampler(Protocol):
    def sample_configs(
        self,
        n: int,
        *,
        fidelity: Mapping[str, Number] | None,
        seed: np.random.Generator,
    ) -> list[Config]: ...
