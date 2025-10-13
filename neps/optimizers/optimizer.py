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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

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
