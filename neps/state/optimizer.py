"""Optimizer state and info dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class BudgetInfo:
    """Information about the budget of an optimizer."""

    max_cost_budget: float | None = None
    used_cost_budget: float = 0.0
    max_evaluations: int | None = None
    used_evaluations: int = 0

    @property
    def remaining_cost_budget(self) -> float | None:
        """The remaining budget."""
        if self.max_cost_budget is None:
            return None
        return self.max_cost_budget - self.used_cost_budget

    def clone(self) -> BudgetInfo:
        return BudgetInfo(
            max_cost_budget=self.max_cost_budget,
            used_cost_budget=self.used_cost_budget,
            max_evaluations=self.max_evaluations,
            used_evaluations=self.used_evaluations,
        )


@dataclass
class OptimizationState:
    """The current state of an optimizer."""

    budget: BudgetInfo | None
    """Information regarind the budget used by the optimization trajectory."""

    shared_state: dict[str, Any]
    """Any information the optimizer wants to store between calls
    to sample and post evaluations.

    For example, an optimizer may wish to store running totals here or various other
    bits of information that may be expensive to recompute.

    Right now there's no support for tensors/arrays and almost no optimizer uses this
    feature. Only cost-cooling uses information out of `.budget`.

    Please reach out to @eddiebergman if you have a use case for this so we can make
    it more robust.
    """


@dataclass
class OptimizerInfo:
    """Meta-information about an optimizer."""

    # TODO(eddiebergman): What are the common keywords
    # we can use that don't have to be crammed into mapping
    info: Mapping[str, Any]
