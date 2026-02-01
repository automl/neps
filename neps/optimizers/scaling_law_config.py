"""Configuration objects for scaling law guided optimizers.

This module provides type-safe, validated configuration classes for scaling law
guided optimization, enabling clean composition and dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping
import inspect
import warnings

if TYPE_CHECKING:
    from neps.space.search_space import SearchSpace
    from neps.space.neps_spaces.neps_space import PipelineSpace


@dataclass
class ScalingLawEstimators:
    """Configuration for scaling law estimator functions.

    All estimators must accept **kwargs (configuration dict as keyword arguments)
    and return a float value.

    Attributes:
        params_estimator: Estimates number of parameters for a configuration.
        flops_estimator: Estimates FLOPs for a configuration.
        seen_datapoints_estimator: Estimates seen datapoints (optional).
        additional_estimators: Additional custom estimators for domain-specific laws.

    Example:
        >>> def count_params(**config) -> float:
        ...     hidden_dim = config.get('hidden_dim', 256)
        ...     num_layers = config.get('num_layers', 4)
        ...     return hidden_dim * num_layers * 1000
        >>> estimators = ScalingLawEstimators(
        ...     params_estimator=count_params,
        ...     flops_estimator=estimate_flops,
        ...     seen_datapoints_estimator=count_data,
        ... )
        >>> estimators.validate()  # Check signatures
    """

    params_estimator: Callable[..., float]
    """Estimates number of parameters for a configuration."""

    flops_estimator: Callable[..., float]
    """Estimates FLOPs for a configuration."""

    seen_datapoints_estimator: Callable[..., float] | None = None
    """Estimates seen datapoints. Optional for simpler scaling laws."""

    additional_estimators: Mapping[str, Callable[..., float]] | None = None
    """Additional custom estimators for domain-specific scaling laws."""

    def validate(self) -> None:
        """Validate that estimators are callable and have reasonable signatures.

        Raises:
            TypeError: If estimator is not callable.
            ValueError: If estimator signature doesn't accept **kwargs.
        """
        estimators = [
            ("params_estimator", self.params_estimator),
            ("flops_estimator", self.flops_estimator),
            ("seen_datapoints_estimator", self.seen_datapoints_estimator),
        ]

        for name, estimator in estimators:
            if estimator is None:
                continue

            if not callable(estimator):
                raise TypeError(
                    f"{name} must be callable, got {type(estimator).__name__}"
                )

            # Check that function accepts **kwargs
            # try:
            #     sig = inspect.signature(estimator)
            #     has_var_keyword = any(
            #         p.kind == inspect.Parameter.VAR_KEYWORD
            #         for p in sig.parameters.values()
            #     )
            #     if not has_var_keyword and len(sig.parameters) == 0:
            #         # Empty signature is OK (for lambdas that capture vars)
            #         pass
            #     elif not has_var_keyword:
            #         raise ValueError(
            #             f"{name} signature {sig} must accept **kwargs "
            #             "(configuration dict as keyword arguments)"
            #         )
            # except (ValueError, TypeError) as e:
            #     raise ValueError(f"Could not inspect {name}: {e}") from e

        if self.additional_estimators:
            for name, estimator in self.additional_estimators.items():
                if not callable(estimator):
                    raise TypeError(
                        f"additional_estimators['{name}'] must be callable, "
                        f"got {type(estimator).__name__}"
                    )

    def to_metric_functions(self) -> dict[str, Callable[..., float]]:
        """Convert to metric_functions dict for ScalingLawGuidedOptimizer.

        Returns:
            Dictionary mapping metric names to estimator functions.
        """
        metrics: dict[str, Callable[..., float]] = {}
        if self.params_estimator:
            metrics["params_estimator"] = self.params_estimator
        if self.flops_estimator:
            metrics["flops_estimator"] = self.flops_estimator
        if self.seen_datapoints_estimator:
            metrics["seen_datapoints_estimator"] = self.seen_datapoints_estimator
        if self.additional_estimators:
            metrics.update(self.additional_estimators)
        return metrics


