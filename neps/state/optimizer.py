"""Optimizer state and info dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neps.state.seed_snapshot import SeedSnapshot


@dataclass
class BudgetInfo:
    """Information about the budget of an optimizer."""

    cost_to_spend: float | None = None
    used_cost_budget: float = 0.0
    max_evaluations: int | None = None
    used_evaluations: int = 0
    fidelities_to_spend: int | float | None = None

    def clone(self) -> BudgetInfo:
        """Create a copy of the budget info."""
        return replace(self)


@dataclass
class OptimizationState:
    """The current state of an optimizer."""

    budget: BudgetInfo | None
    """Information regarind the budget used by the optimization trajectory."""

    seed_snapshot: SeedSnapshot
    """The state of the random number generators at the time of the last sample."""

    shared_state: dict[str, Any] | None
    """Any information the optimizer wants to store between calls
    to sample and post evaluations.

    For example, an optimizer may wish to store running totals here or various other
    bits of information that may be expensive to recompute.

    Right now there's no support for tensors/arrays and almost no optimizer uses this
    feature. Only cost-cooling uses information out of `.budget`.

    Please reach out to @eddiebergman if you have a use case for this so we can make
    it more robust.
    """
    worker_ids: list[str] | None = None
    """The list of workers that have been created so far."""
