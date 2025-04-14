"""Implements functionallity for the priorband sampling strategy."""

from __future__ import annotations

import pprint
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from neps.optimizers.utils import brackets
from neps.sampling import Prior, Sampler
from neps.space import ConfigEncoder

if TYPE_CHECKING:
    import pandas as pd

    from neps.space.parameters import Parameter


@dataclass
class PriorBandSampler:
    """A Sampler implementing the PriorBand algorithm for sampling.

    * https://openreview.net/forum?id=uoiwugtpCH&noteId=xECpK2WH6k
    """

    parameters: Mapping[str, Parameter]
    """The parameters to consider."""

    encoder: ConfigEncoder
    """The encoder to use for encoding and decoding configurations into tensors."""

    mutation_rate: float
    """The mutation rate to use when sampling from the incumbent distribution."""

    mutation_std: float
    """The mutation deviation to use when sampling from the incumbent distribution."""

    eta: int
    """The eta value to use for the SH bracket."""

    early_stopping_rate: int
    """The early stopping rate to use for the SH bracket."""

    fid_bounds: tuple[int, int] | tuple[float, float]
    """The fidelity bounds."""

    def sample_config(self, table: pd.DataFrame, rung: int) -> dict[str, Any]:
        """Samples a configuration using the PriorBand algorithm.

        Args:
            table: The table of all the trials that have been run.
            rung_to_sample_for: The rung to sample for.

        Returns:
            The sampled configuration.
        """
        rung_to_fid, rung_sizes = brackets.calculate_sh_rungs(
            bounds=self.fid_bounds,
            eta=self.eta,
            early_stopping_rate=self.early_stopping_rate,
        )
        max_rung = max(rung_sizes)

        prior_dist = Prior.from_parameters(self.parameters)

        pprint.pp(prior_dist)
        pprint.pp(self.parameters)

        # Below we will follow the "geomtric" spacing
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
                    config = prior_dist.sample_config(to=self.encoder)
                case "random":
                    _sampler = Sampler.uniform(ndim=self.encoder.ndim)
                    config = _sampler.sample_config(to=self.encoder)

            return config

        # Otherwise, we now further split the `prior` weight into `(prior, inc)`

        # 1. Select the top `1//eta` percent of configs at the highest rung supporting it
        rungs_with_at_least_eta = rung_counts[rung_counts >= self.eta].index  # type: ignore
        rung_table: pd.DataFrame = completed[  # type: ignore
            completed.index.get_level_values("rung") == rungs_with_at_least_eta.max()
        ]

        K = len(rung_table) // self.eta
        top_k_configs = rung_table.nsmallest(K, columns=["perf"])["config"].tolist()

        # 2. Get the global incumbent, and build a prior distribution around it
        inc = completed.loc[completed["perf"].idxmin()]["config"]
        inc_dist = Prior.from_parameters(self.parameters, center_values=inc)

        # 3. Calculate a ratio score of how likely each of the top K configs are under
        # the prior and inc distribution, weighing them by their position in the top K
        weights = torch.arange(K, 0, -1)

        top_k_pdf_inc = inc_dist.pdf_configs(top_k_configs, frm=self.encoder)  # type: ignore
        top_k_pdf_prior = prior_dist.pdf_configs(top_k_configs, frm=self.encoder)  # type: ignore

        unnormalized_inc_score = (weights * top_k_pdf_inc).sum()
        unnormalized_prior_score = (weights * top_k_pdf_prior).sum()
        total_score = unnormalized_inc_score + unnormalized_prior_score

        inc_ratio = float(unnormalized_inc_score / total_score)
        prior_ratio = float(unnormalized_prior_score / total_score)

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
                return prior_dist.sample_config(to=self.encoder)
            case "random":
                _sampler = Sampler.uniform(ndim=self.encoder.ndim)
                return _sampler.sample_config(to=self.encoder)
            case "inc":
                assert inc is not None
                return mutate_config(
                    inc,
                    parameters=self.parameters,
                    mutation_rate=self.mutation_rate,
                    std=self.mutation_std,
                    seed=None,
                )

        raise RuntimeError(f"Unknown policy: {policy}")


def mutate_config(
    config: dict[str, Any],
    parameters: Mapping[str, Parameter],
    *,
    mutation_rate: float = 0.5,
    std: float = 0.25,
    seed: torch.Generator | None = None,
) -> dict[str, Any]:
    if seed is not None:
        raise NotImplementedError("Seed is not implemented yet.")

    # This prior places a guassian on the numericals and places a 0 probability on the
    # current value of the categoricals.
    mutatation_prior = Prior.from_parameters(
        parameters,
        center_values=config,
        # Assign a confidence of 0 to our current categoricals
        # to ensure they dont get sampled
        confidence_values={
            key: 0 if hp.domain.is_categorical else (1 - std)
            for key, hp in parameters.items()
        },
    )
    config_encoder = ConfigEncoder.from_parameters(parameters)

    mutant: dict[str, Any] = mutatation_prior.sample_config(to=config_encoder)
    mutatant_selection = torch.rand(len(config), generator=seed) < mutation_rate

    return {
        key: mutant[key] if select_mutant else config[key]
        for key, select_mutant in zip(mutant.keys(), mutatant_selection, strict=False)
    }
