"""Optimizer interface.

By implementing the [`AskFunction`][neps.optimizers.optimizer.AskFunction] protocol,
you can inject your own optimizer into the neps runtime.

```python
class MyOpt:

    def __init__(self, space: SearchSpace, ...): ...

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]: ...

neps.run(..., optimizer=MyOpt)

# Or with optimizer hyperparameters
neps.run(..., optimizer=(MyOpt, {"a": 1, "b": 2}))
```
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

# `typing.TypedDict` does not pick up `NotRequired` through PEP-563 string
# annotations, so we take both from `typing_extensions`.
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.pipeline_eval import UserResultDict
    from neps.state.trial import Trial


class OptimizerInfo(TypedDict):
    """Information about the optimizer, usually used for serialization."""

    name: str
    """The name of the optimizer."""

    info: Mapping[str, Any]
    """Additional information about the optimizer.

    Usually this will be the keyword arguments used to initialize the optimizer.
    """

    fidelity_name: NotRequired[str | None]
    """The key a fidelity takes in a trial's config, if the space has one.

    Resolved once at the start of a run and persisted, so that anything
    summarizing the run later can account for fidelities without having to
    reconstruct the optimizer. Absent for runs created before this was stored,
    and ignored when checking that a resumed run matches its optimizer.
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

        This is called whenever a trial is added to the neps state.

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
        (figures, logs, metadata, etc) that the neps runtime will handle for persistence.
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
            after every evaluated trial, when running `neps.run(..., live_plots=True)`.

        Example:
            >>> from neps.optimizers.optimizer import Artifact, ArtifactType
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
