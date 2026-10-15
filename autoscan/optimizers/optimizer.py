"""Optimizer interface.

By implementing the [`AskFunction`][autoscan.optimizers.optimizer.AskFunction] protocol,
you can inject your own optimizer into the autoscan runtime.

```python
class MyOpt:

    def __init__(self, space: SearchSpace, ...): ...

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]: ...

autoscan.run(..., optimizer=MyOpt)

# Or with optimizer hyperparameters
autoscan.run(..., optimizer=(MyOpt, {"a": 1, "b": 2}))
```
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from autoscan.state.optimizer import BudgetInfo
    from autoscan.state.pipeline_eval import UserResultDict
    from autoscan.state.trial import Trial
    from autoscan.utils.cost_estimator import CostEstimator


class OptimizerInfo(TypedDict):
    """Information about the optimizer, usually used for serialization."""

    name: str
    """The name of the optimizer."""

    info: Mapping[str, Any]
    """Additional information about the optimizer.

    Usually this will be the keyword arguments used to initialize the optimizer.
    """

    derived: NotRequired[Mapping[str, Any]]
    """Values the optimizer computed from its settings, e.g. the rung layout of
    multi-fidelity optimizers. Informational only, not compared when resuming a run.
    """


class ArtifactType(Enum):
    """Supported artifact types for optimizer results."""

    TEXT = "text"
    FIGURE = "figure"
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    BYTES = "bytes"


@dataclass
class Artifact:
    """Single artifact produced by optimizer.

    Attributes:
        name: Unique identifier for this artifact.
        content: The actual artifact content (string, Figure, dict, bytes, etc).
        artifact_type: Type of artifact, determines how it will be persisted.
        metadata: Optional metadata about the artifact (e.g., format, dimensions).
    """

    name: str
    content: Any
    artifact_type: ArtifactType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampledConfig:
    id: str
    config: Mapping[str, Any]
    previous_config_id: str | None = None


@dataclass
class ImportedConfig:
    id: str
    config: Mapping[str, Any]
    result: UserResultDict


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
            n: The number of configurations to sample. If you do not support
                sampling multiple configurations at once, you should raise
                a `ValueError`.

        Returns:
            The sampled configuration(s)
        """
        ...

    @abstractmethod
    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        """Add a trial to the optimizer's internal state.

        This is called whenever a trial is added to the autoscan state.

        Args:
            external_evaluations: The configs to add.
            trials: All of the trials that are known about.
        """
        ...

    @abstractmethod
    def get_trial_artifacts(
        self,
        trials: Mapping[str, Trial] | None = None,
    ) -> list[Artifact] | None:
        """Return artifacts for the runtime to persist.

        This is an optional method that optimizers can implement to return artifacts
        (figures, logs, metadata, etc) that the autoscan runtime will handle for
        persistence.
        The runtime takes responsibility for all I/O operations.

        Args:
            trials: All evaluated trials, passed by runtime for context.
                Optional - optimizers can ignore if not needed.

        Returns:
            List of Artifact objects to persist, or None if no artifacts should be
            persisted.

        Note:
            This method is optional. Optimizers that don't need to persist artifacts
            should return None (the default behavior). The runtime only calls it,
            after every evaluated trial, when running `autoscan.run(...,
            live_plots=True)`.

        Example:
            >>> from autoscan.optimizers.optimizer import Artifact, ArtifactType
            >>> def get_trial_artifacts(self, trials=None) -> list[Artifact] | None:
            ...     import matplotlib.pyplot as plt
            ...     fig, ax = plt.subplots()
            ...
            ...     # Can use trials to generate more meaningful artifacts
            ...     if trials:
            ...         results = [
            ...             t.report.objective_to_minimize for t in trials.values()
            ...         ]
            ...         ax.plot(results)
            ...
            ...     return [
            ...         Artifact("loss_curve", fig, ArtifactType.FIGURE),
            ...         Artifact("best_config", self.best_config, ArtifactType.JSON),
            ...     ]
        """
        return None


class ScalingOptimizer:
    """Base class for optimizers that are ScAn native.

    `autoscan.run(..., cost_estimator=..., max_cost_per_config=...)` hands both to
    every optimizer inheriting from this class, right after it was created.

    ```python
    class MyOpt(ScalingOptimizer):
        def __call__(self, trials, budget_info, n=None):
            if self.cost_estimator is not None:
                cost = self.cost_estimator(**config)
            ...
    ```
    """

    cost_estimator: CostEstimator | None = None
    """Estimates the cost of a configuration, `None` if the run has none."""

    max_cost_per_config: float | None = None
    """Configurations estimated to cost more than this should not be sampled."""

    def set_cost_estimator(
        self,
        cost_estimator: CostEstimator | None,
        *,
        max_cost_per_config: float | None = None,
    ) -> None:
        """Called by `autoscan.run()` with the cost estimator of the run."""
        self.cost_estimator = cost_estimator
        self.max_cost_per_config = max_cost_per_config

    def budget_constraint(
        self,
        constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
        *,
        constants: Mapping[str, Any] | None = None,
    ) -> Callable[[Mapping[str, Any]], float] | None:
        """Combine `max_cost_per_config` with a user given constraint.

        Args:
            constraints_func: A constraint that is `>= 0` for valid configurations.
            constants: Values to add to a configuration before estimating its cost
                and evaluating `constraints_func`, for optimizers that check
                configurations without their constants.

        Returns:
            A function that is `>= 0` for configurations that satisfy the constraints
        """
        estimator, max_cost = self.cost_estimator, self.max_cost_per_config
        has_budget = estimator is not None and max_cost is not None
        if not has_budget and not constants:
            return constraints_func
        if not has_budget and constraints_func is None:
            return None

        def _constraint(config: Mapping[str, Any]) -> float:
            # Both the cost estimator and the user constraint see the full config
            full_config = {**(constants or {}), **config}
            values = []
            if has_budget:
                # Normalize the cost to a slack value between 0 and 1.
                values.append(1.0 - estimator(**full_config) / max_cost)  # type: ignore
            if constraints_func is not None:
                values.append(constraints_func(full_config))
            return min(values)

        return _constraint
