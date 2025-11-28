from __future__ import annotations

from typing import TypeAlias

import numpy as np


class LinearScalarization:
    """A utility class for randomly weighted linear scalarization
    of multiple objectives."""

    def __init__(
        self,
        scalarization_weights: list[float] | dict[str, float] | None = None,
    ) -> None:
        """Initialize the linear scalarization class."""
        if isinstance(scalarization_weights, dict):
            scalarization_weights = list(scalarization_weights.values())
        self.scalarization_weights = scalarization_weights

    def scalarize(
        self,
        objective_values: list[float],
    ) -> float:
        """Scalarize the given objectives using randomly sampled weights.

        Args:
            objective_values: The list of objective values to scalarize.

        Returns:
            The scalarized objective value.
        """

        if self.scalarization_weights is None:
            self.scalarization_weights = np.random.uniform(size=len(objective_values))
            self.scalarization_weights /= np.sum(self.scalarization_weights)

        return np.dot(objective_values, self.scalarization_weights)


Scalarization: TypeAlias = LinearScalarization
