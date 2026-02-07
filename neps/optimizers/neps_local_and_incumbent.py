from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Tuple

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

    random_ratio: float = 0.0
    """The ratio of random sampling vs incumbent sampling."""

    local_prior: dict[str, Any] | None = None
    """The local prior configuration."""

    inc_takeover_mode: Literal[0, 1, 2, 3] = 0
    """The incumbent takeover mode.
    0: Always mutate the first config.
    1: Use the global incumbent.
    2: Crossover between global incumbent and first config.
    3: Choose randomly between 0, 1, and 2.
    """

    mutation_mode: Tuple[Literal["random", "fixed"], float] = ("random", 0.5)
    """The mutation mode.
    ("random", ratio): Mutate a random number of parameters up to the given ratio.
    ("fixed", n): Mutate a fixed number of parameters given by n.
    """

    def sample_config(self, table: pd.DataFrame) -> dict[str, Any]:  # noqa: C901
        """Sample a configuration based on the PriorBand algorithm.

        Args:
            table (pd.DataFrame): The table containing the configurations and their
                performance.

        Returns:
            dict[str, Any]: A sampled configuration.
        """

        completed: pd.DataFrame = table[table["perf"].notna()]  # type: ignore
        if completed.empty:
            logging.warning("No local prior found. Sampling randomly from the space.")
            return (
                self.local_prior
                if self.local_prior is not None
                else self._sample_random()
            )

        # If no local prior is given, save the first config as the local prior
        if self.local_prior is None:
            first_config = completed.iloc[0]["config"]
            assert isinstance(first_config, dict)
            self.local_prior = first_config

        # Get the incumbent configuration
        inc_config = completed.loc[completed["perf"].idxmin()]["config"]
        first_config = self.local_prior
        assert isinstance(inc_config, dict)

        # Decide whether to sample randomly or from the incumbent
        if random.random() < self.random_ratio:
            return self._sample_random()

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

    def _sample_random(self) -> dict[str, Any]:
        # Sample randomly from the space
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.upper

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=sampling.RandomSampler({}),
            environment_values=_environment_values,
        )
        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def _mutate_inc(self, inc_config: dict[str, Any]) -> dict[str, Any]:
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.upper

        if self.mutation_mode[0] == "random":
            assert 0 < self.mutation_mode[1] <= 1
            n_mutations = random.randint(1, int(len(inc_config) * self.mutation_mode[1]))
        elif self.mutation_mode[0] == "fixed":
            assert self.mutation_mode[1] > 0
            n_mutations = min(len(inc_config), self.mutation_mode[1])
        else:
            raise ValueError(f"Invalid mutation mode: {self.mutation_mode[0]}")

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=neps.space.neps_spaces.sampling.MutatateUsingCentersSampler(
                predefined_samplings=inc_config,
                n_mutations=n_mutations,
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
