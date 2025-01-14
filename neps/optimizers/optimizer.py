"""Optimizer interface."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


@dataclass
class SampledConfig:
    id: str
    config: Mapping[str, Any]
    previous_config_id: str | None = None


class AskFunction(Protocol):
    """Interface to implement the ask of optimizer."""

    @abstractmethod
    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        """Sample a new configuration.

        Args:
            trials: All of the trials that are known about.
            budget_info: information about the budget constraints.

        Returns:
            The sampled configuration(s)
        """
        ...
