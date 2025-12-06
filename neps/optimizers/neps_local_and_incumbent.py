from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space, sampling

if TYPE_CHECKING:
    import pandas as pd

    from neps.space.neps_spaces.parameters import PipelineSpace


@dataclass
class NePSLocalPriorIncumbentSampler:
    """Implement a sampler that samples from the incumbent."""

    space: PipelineSpace
    """The pipeline space to optimize over."""

    local_prior: dict[str, Any]
    """The first config to sample."""

    inc_takeover_mode: Literal[0, 1, 2, 3] = 0
    """The incumbent takeover mode.
    0: Always mutate the first config.
    1: Use the global incumbent.
    2: Crossover between global incumbent and first config.
    3: Choose randomly between 0, 1, and 2.
    """

    def sample_config(self, table: pd.DataFrame) -> dict[str, Any]:
        """Sample a configuration based on the PriorBand algorithm.

        Args:
            table (pd.DataFrame): The table containing the configurations and their
                performance.
            rung (int): The current rung of the optimization.

        Returns:
            dict[str, Any]: A sampled configuration.
        """

        completed: pd.DataFrame = table[table["perf"].notna()]  # type: ignore
        if completed.empty:
            return self.local_prior

        # Get the incumbent configuration
        inc_config = completed.loc[completed["perf"].idxmin()]["config"]
        first_config = self.local_prior
        assert isinstance(inc_config, dict)

        match self.inc_takeover_mode:
            case 0:
                # Always mutate the first config.
                new_config = self._mutate_inc(inc_config=first_config)
            case 1:
                # Use the global incumbent.
                new_config = self._mutate_inc(inc_config=inc_config)
            case 2:
                # Crossover between global incumbent and first config.
                new_config = self._crossover_incs(
                    inc_config=inc_config,
                    first_config=first_config,
                )
            case 3:
                # Choose randomly between 0, 1, and 2.
                match random.randint(0, 2):
                    case 0:
                        new_config = self._mutate_inc(inc_config=first_config)
                    case 1:
                        new_config = self._mutate_inc(inc_config=inc_config)
                    case 2:
                        new_config = self._crossover_incs(
                            inc_config=inc_config,
                            first_config=first_config,
                        )
                    case _:
                        raise ValueError(
                            "This should never happen. Only for type checking."
                        )
            case _:
                raise ValueError(f"Invalid inc_takeover_mode: {self.inc_takeover_mode}")
        return new_config

    def _mutate_inc(self, inc_config: dict[str, Any]) -> dict[str, Any]:
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.upper

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=neps.space.neps_spaces.sampling.MutatateUsingCentersSampler(
                predefined_samplings=inc_config,
                n_mutations=max(1, random.randint(1, int(len(inc_config) / 2))),
            ),
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def _crossover_incs(
        self, inc_config: dict[str, Any], first_config: dict[str, Any]
    ) -> dict[str, Any]:
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.upper

        # Crossover between the best two trials' configs to create a new config.
        try:
            crossover_sampler = sampling.CrossoverByMixingSampler(
                predefined_samplings_1=inc_config,
                predefined_samplings_2=first_config,
                prefer_first_probability=0.5,
            )
            _resolved_pipeline, resolution_context = neps_space.resolve(
                pipeline=self.space,
                domain_sampler=crossover_sampler,
                environment_values=_environment_values,
            )
        except sampling.CrossoverNotPossibleError:
            # A crossover was not possible for them. Increase configs and try again.
            # If we have tried all crossovers, mutate the best instead.
            # Mutate 50% of the top trial's config.
            _resolved_pipeline, resolution_context = neps_space.resolve(
                pipeline=self.space,
                domain_sampler=sampling.MutatateUsingCentersSampler(
                    predefined_samplings=inc_config,
                    n_mutations=max(1, int(len(inc_config) / 2)),
                ),
                environment_values=_environment_values,
            )
        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)
