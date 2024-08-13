# TODO: Placeholder class
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from neps.utils.types import Number


# TODO: Placeholder...
@dataclass(frozen=True)
class Config:
    values: Mapping[str, Any]
    fidelity: Mapping[str, Number] | None

    @classmethod
    def new(
        cls,
        values: Mapping[str, Any],
        fidelity: Mapping[str, Number] | None = None,
    ) -> Config:
        return Config(values=values, fidelity=fidelity)

    def clone(self, **changes: Any) -> Config:
        return replace(self, **changes)
