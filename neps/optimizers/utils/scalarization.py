from __future__ import annotations

from typing import TypeAlias

import numpy as np

from neps.optimizers.bayesian_optimization import _get_reference_point


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


class HypervolumeScalarization:
    """A utility class for hypervolume-based scalarization
    of multiple objectives."""

    def __init__(
        self,
        scalarization_weights: list[float] | dict[str, float] | None = None,
        reference_point: list[float] | None = None,
    ) -> None:
        """Initialize the hypervolume scalarization class."""
        if isinstance(scalarization_weights, dict):
            scalarization_weights = list(scalarization_weights.values())
        self.scalarization_weights = scalarization_weights
        self.reference_point = reference_point

    def scalarize(
        self,
        objective_values: list[float],
        all_obj_vals: list[list[float]],
    ) -> float:
        """Scalarize the given objectives using hypervolume-based method.

        Args:
            objective_values: The list of objective values to scalarize.

        Returns:
            The scalarized objective value.
        """
        objective_vals = np.array(objective_values)
        if self.reference_point is None:
            self.reference_point = _get_reference_point(
                np.array(all_obj_vals),
            )

        shifted_objectives = self.reference_point - objective_vals

        if self.scalarization_weights is None:
            scalar_wts = np.random.uniform(size=len(objective_vals))
            scalar_wts /= np.sum(scalar_wts)
        else:
            scalar_wts = np.array(self.scalarization_weights)

        product = (1.0 / scalar_wts) * shifted_objectives

        return np.power(np.min(product, axis=-1), len(objective_vals))


Scalarization: TypeAlias = LinearScalarization | HypervolumeScalarization
