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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
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
