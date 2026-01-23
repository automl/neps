from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space, sampling

if TYPE_CHECKING:
    import neps.state.optimizer as optimizer_state
    import neps.state.trial as trial_state
    from neps.space.neps_spaces.parameters import PipelineSpace
    from neps.state.pipeline_eval import UserResultDict
    from neps.state.trial import Trial


@dataclass
class NePSExhaustiveGridSearch:
    """Implement a sampler that samples from the incumbent."""

    space: PipelineSpace
    """The pipeline space to optimize over."""

    sampling_density: int = 10

    sampling_tries: int = 100
    """The number of tries to sample a new configuration."""

    def _sample_random(
        self,
    ):
        # Sample randomly from the space
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.upper

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=sampling.GridSampler(),
            environment_values=_environment_values,
        )
        return _resolved_pipeline, resolution_context

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> dict[str, Any]:  # noqa: C901
        """Sample a configuration based on the PriorBand algorithm.

        Args:
            table (pd.DataFrame): The table containing the configurations and their
                performance.

        Returns:
            dict[str, Any]: A sampled configuration.
        """
        n_prev_trials = len(trials)
        n_requested = 1 if n is None else n


        for _ in range(self.sampling_tries):
            new_config = self._sample_random()
            if neps_space.NepsCompatConverter.to_neps_config(new_config[1]) not in [
                trial.config for trial in trials.values()
            ]:
                return neps_space._prepare_sampled_configs(
                    [new_config], n_prev_trials, n_requested == 1
                )

        raise RuntimeError(
            f"Failed to sample a new configuration after {self.sampling_tries} tries."
        )
