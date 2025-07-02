"""PriorBand Sampler for NePS Optimizers.
This sampler implements the PriorBand algorithm, which is a sampling strategy
that combines prior knowledge with random sampling to efficiently explore the search
space. It uses a combination of prior sampling, incumbent mutation, and random sampling
based on the fidelity bounds and SH bracket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import neps.space.neps_spaces.parameters
import neps.space.neps_spaces.sampling
from neps.optimizers.utils import brackets
from neps.space.neps_spaces import neps_space

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PriorBandSampler:
    """Implement a sampler based on PriorBand."""

    """The pipeline space to optimize over."""
    space: neps.space.neps_spaces.parameters.Pipeline

    """The eta value to use for the SH bracket."""
    eta: int

    """The early stopping rate to use for the SH bracket."""
    early_stopping_rate: int

    """The fidelity bounds."""
    fid_bounds: tuple[int, int] | tuple[float, float]

    def sample_config(self, table: pd.DataFrame, rung: int) -> dict[str, Any]:
        """Sample a configuration based on the PriorBand algorithm.

        Args:
            table (pd.DataFrame): The table containing the configurations and their
            performance.
            rung (int): The current rung of the optimization.

        Returns:
            dict[str, Any]: A sampled configuration.
        """
        rung_to_fid, rung_sizes = brackets.calculate_sh_rungs(
            bounds=self.fid_bounds,
            eta=self.eta,
            early_stopping_rate=self.early_stopping_rate,
        )
        max_rung = max(rung_sizes)

        # Below we will follow the "geometric" spacing
        w_random = 1 / (1 + self.eta**rung)
        w_prior = 1 - w_random

        completed: pd.DataFrame = table[table["perf"].notna()]  # type: ignore

        # To see if we activate incumbent sampling, we check:
        # 1) We have at least one fully complete run
        # 2) We have spent at least one full SH bracket worth of fidelity
        # 3) There is at least one rung with eta evaluations to get the top 1/eta configs
        completed_rungs = completed.index.get_level_values("rung")
        one_complete_run_at_max_rung = (completed_rungs == max_rung).any()

        # For SH bracket cost, we include the fact we can continue runs,
        # i.e. resources for rung 2 discounts the cost of evaluating to rung 1,
        # only counting the difference in fidelity cost between rung 2 and rung 1.
        cost_per_rung = {
            i: rung_to_fid[i] - rung_to_fid.get(i - 1, 0) for i in rung_to_fid
        }

        cost_of_one_sh_bracket = sum(rung_sizes[r] * cost_per_rung[r] for r in rung_sizes)
        current_cost_used = sum(r * cost_per_rung[r] for r in completed_rungs)
        spent_one_sh_bracket_worth_of_fidelity = (
            current_cost_used >= cost_of_one_sh_bracket
        )

        # Check that there is at least rung with `eta` evaluations
        rung_counts = completed.groupby("rung").size()
        any_rung_with_eta_evals = (rung_counts == self.eta).any()

        # If the conditions are not met, we sample from the prior or randomly depending on
        # the geometrically distributed prior and uniform weights
        if (
            one_complete_run_at_max_rung is False
            or spent_one_sh_bracket_worth_of_fidelity is False
            or any_rung_with_eta_evals is False
        ):
            policy = np.random.choice(["prior", "random"], p=[w_prior, w_random])
            match policy:
                case "prior":
                    return self._sample_prior()
                case "random":
                    return self._sample_random()
                case _:
                    raise RuntimeError(f"Unknown policy: {policy}")

        # Otherwise, we now further split the `prior` weight into `(prior, inc)`

        # 1. Select the top `1//eta` percent of configs at the highest rung supporting it
        rungs_with_at_least_eta = rung_counts[rung_counts >= self.eta].index  # type: ignore
        rung_table: pd.DataFrame = completed[  # type: ignore
            completed.index.get_level_values("rung") == rungs_with_at_least_eta.max()
        ]

        K = len(rung_table) // self.eta
        rung_table.nsmallest(K, columns=["perf"])["config"].tolist()

        # 2. Get the global incumbent
        inc_config = completed.loc[completed["perf"].idxmin()]["config"]

        # 3. Calculate a ratio score of how likely each of the top K configs are under
        # TODO: [lum]: Here I am simply using fixed values.
        #  Will maybe have to come up with a way to approximate the pdf for the top
        # configs.
        inc_ratio = 0.9
        prior_ratio = 0.1

        # 4. And finally, we distribute the original w_prior according to this ratio
        w_inc = w_prior * inc_ratio
        w_prior = w_prior * prior_ratio
        assert np.isclose(w_prior + w_inc + w_random, 1.0)

        # Now we use these weights to choose which sampling distribution to sample from
        policy = np.random.choice(
            ["prior", "inc", "random"],
            p=[w_prior, w_inc, w_random],
        )
        match policy:
            case "prior":
                return self._sample_prior()
            case "random":
                return self._sample_random()
            case "inc":
                assert inc_config is not None
                return self._mutate_inc(inc_config)
        raise RuntimeError(f"Unknown policy: {policy}")

    def _sample_prior(self) -> dict[str, Any]:
        # TODO: [lum] have a CenterSampler as fallback, not Random
        _try_always_priors_sampler = (
            neps.space.neps_spaces.sampling.PriorOrFallbackSampler(
                fallback_sampler=neps.space.neps_spaces.sampling.RandomSampler(
                    predefined_samplings={}
                ),
                prior_use_probability=1,
            )
        )

        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.max_value

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=_try_always_priors_sampler,
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def _sample_random(self) -> dict[str, Any]:
        _environment_values = {}
        _fidelity_attrs = self.space.fidelity_attrs
        for fidelity_name, fidelity_obj in _fidelity_attrs.items():
            _environment_values[fidelity_name] = fidelity_obj.max_value

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=neps.space.neps_spaces.sampling.RandomSampler(
                predefined_samplings={}
            ),
            environment_values=_environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)

    def _mutate_inc(self, inc_config: dict[str, Any]) -> dict[str, Any]:
        data = neps_space.NepsCompatConverter.from_neps_config(config=inc_config)

        _resolved_pipeline, resolution_context = neps_space.resolve(
            pipeline=self.space,
            domain_sampler=neps.space.neps_spaces.sampling.MutatateUsingCentersSampler(
                predefined_samplings=data.predefined_samplings,
                n_mutations=max(1, random.randint(1, int(len(inc_config) / 2))),
            ),
            environment_values=data.environment_values,
        )

        config = neps_space.NepsCompatConverter.to_neps_config(resolution_context)
        return dict(**config)
