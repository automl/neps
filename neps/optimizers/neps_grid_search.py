"""V6: Dynamic Queue-Based Exhaustive Grid Search.

This module implements a completely dynamic approach to exhaustive grid search that
discovers the search space structure during sampling rather than pre-analyzing it.
The sampler maintains a queue of sampling decisions and systematically enumerates
all combinations by incrementing indices in a counter-like fashion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from neps.optimizers.optimizer import SampledConfig
from neps.space.neps_spaces.neps_space import NepsCompatConverter, resolve
from neps.space.neps_spaces.sampling import QueueBasedSampler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import neps.state.optimizer as optimizer_state
    import neps.state.trial as trial_state
    from neps.space.neps_spaces.parameters import PipelineSpace


@dataclass
class NePSExhaustiveGridSearch:
    """Dynamic queue-based exhaustive grid search optimizer.

    This optimizer discovers the search space structure dynamically during sampling.
    It maintains a queue of tuples (param_type, max_value, current_index) representing
    each sampling decision. The queue starts empty and grows as new parameters are
    discovered through conditional branches.

    After each configuration is sampled, the queue is incremented like a multi-base
    counter with cascading carries. When the queue becomes empty, all configurations
    have been generated.

    Args:
        pipeline_space: The pipeline space to search.
        sampling_density: Number of grid points for numerical parameters (default: 5).
    """

    pipeline_space: PipelineSpace
    sampling_density: int = 5
    ignore_fidelity: bool | Literal["highest_fidelity"] = False

    # These fields are initialized in __post_init__, not by the user
    queue: list[tuple[str, int, int]] = field(default_factory=list, init=False)
    sampler: QueueBasedSampler = field(init=False)
    fidelity_values: dict[str, Any] = field(default_factory=dict, init=False)
    configs_generated: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize the queue, sampler, and fidelity values."""
        self.queue = []
        self.sampler = QueueBasedSampler(
            queue=self.queue, sampling_density=self.sampling_density
        )

        # Extract fidelity values (use middle value)
        self.fidelity_values = self.sampled_fidelity_values()
        self.configs_generated = 0
        self._done = False

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig:
        """Generate the next configuration.

        Returns:
            A SampledConfig object containing the next configuration.

        Raises:
            StopIteration: When all configurations have been generated (queue is empty
                after incrementing).
        """

        if n is not None and n > 1:
            raise ValueError("NePSExhaustiveGridSearch only supports n=1")

        if self._done:
            raise StopIteration("All configurations have been generated")

        # Generate config ID based on existing trials
        max_trial_id = max((int(tid) for tid in trials), default=0)
        config_id = str(max_trial_id + 1)

        # Reset sampler position for new config
        self.sampler.reset_position()

        # Sample configuration (may extend queue if new paths discovered)
        _resolved, resolution_context = resolve(
            pipeline=self.pipeline_space,
            domain_sampler=self.sampler,
            environment_values=self.fidelity_values,
        )

        # Convert to config dict
        config = NepsCompatConverter.to_neps_config(resolution_context=resolution_context)

        # Increment queue after successful sampling
        if not self._increment_queue():
            # Queue is exhausted - no more configs after this one
            self._done = True

        self.configs_generated += 1
        return SampledConfig(config=config, id=config_id, previous_config_id=None)

    def sampled_fidelity_values(self) -> dict[str, Any]:
        """Sample fidelity values based on the pipeline's fidelity attributes.

        Returns:
            A dictionary mapping fidelity names to their sampled values.
        """
        environment_values = {}
        fidelity_attrs = self.pipeline_space.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            if self.ignore_fidelity == "highest_fidelity":
                environment_values[fidelity_name] = fidelity_obj.upper
            elif not self.ignore_fidelity:
                raise ValueError(
                    "RandomSearch does not support fidelities by default. Consider"
                    " using a different optimizer or setting `ignore_fidelity=True` or"
                    " `highest_fidelity`."
                )
            else:
                raise ValueError(
                    f"Unsupported value for ignore_fidelity: {self.ignore_fidelity}"
                )
        return environment_values

    def _increment_queue(self) -> bool:
        """Increment the queue counter with cascading carries.

        This method increments indices like a multi-base counter. Starting from the
        last element, it increments the current_index. If the index reaches max_value,
        that element is removed (exhausted) and the cascade continues to the previous
        element. If any element is successfully incremented without exhausting it,
        the method returns True. If all elements are exhausted and the queue becomes
        empty, it returns False.

        Returns:
            True if there are more configurations to generate, False if the queue
            is exhausted.
        """
        if not self.queue:
            return False

        # Start from the last element
        idx = len(self.queue) - 1

        while idx >= 0:
            param_type, max_value, current_index = self.queue[idx]
            new_index = current_index + 1

            if new_index < max_value:
                # Still have room in this position
                self.queue[idx] = (param_type, max_value, new_index)
                return True
            # This position is exhausted, remove it and cascade
            self.queue.pop(idx)
            idx -= 1

        # All positions exhausted
        return False


__all__ = ["NePSExhaustiveGridSearch"]
