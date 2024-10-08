"""Optimizer state and info dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class BudgetInfo:
    """Information about the budget of an optimizer."""

    max_cost_budget: float | None = None
    used_cost_budget: float = 0.0
    max_evaluations: int | None = None
    used_evaluations: int = 0

    def clone(self) -> BudgetInfo:
        """Create a copy of the budget info."""
        return replace(self)


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
