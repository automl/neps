"""Implements the priorband sampling strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from neps.optimizers.utils import brackets
from neps.sampling.priors import Prior
from neps.sampling.samplers import Sampler
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    import pandas as pd

    from neps.search_spaces.search_space import SearchSpace


@dataclass
class PriorBandArgs:
    """Arguments for the PriorBand sampler.

    Args:
        mutation_rate: The mutation rate for the PriorBand algorithm when sampling
            from the incumbent.
        mutation_std: The standard deviation for the mutation rate when sampling
            from the incumbent.
    """

    name: ClassVar = "priorband"

    mutation_rate: float
    mutation_std: float


def mutate_config(
    config: dict[str, Any],
    space: SearchSpace,
    *,
    mutation_rate: float = 0.5,
    std: float = 0.25,
    include_fidelity: bool = False,
    seed: torch.Generator | None = None,
) -> dict[str, Any]:
    if seed is not None:
        raise NotImplementedError("Seed is not implemented yet.")

    parameters = {**space.numerical, **space.categoricals}

    # Assign a confidence of 0 to our current categoricals to ensure they dont get sampled
    confidence_values = {
        key: 0 if hp.domain.is_categorical else (1 - std)
        for key, hp in parameters.items()
    }

    # This prior places a guassian on the numericals and places a 0 probability on the
    # current value of the categoricals.
    mutate_prior = Prior.from_config(
        config,
        space=space,
        confidence_values=confidence_values,
        include_fidelity=include_fidelity,
    )
    config_encoder = ConfigEncoder.from_space(space, include_fidelity=include_fidelity)

    mutant: dict[str, Any] = mutate_prior.sample_config(to=config_encoder)
    mutatant_selection = torch.rand(len(config), generator=seed) < mutation_rate

    return {
        key: mutant[key] if select_mutant else config[key]
        for key, select_mutant in zip(mutant.keys(), mutatant_selection, strict=False)
    }


def sample_with_priorband(
    *,
    table: pd.DataFrame,
    rung_to_sample_for: int,
    # Search Space
    space: SearchSpace,
    encoder: ConfigEncoder,
    # Inc sampling params
    inc_mutation_rate: float,
    inc_mutation_std: float,
    # SH parameters to calculate the rungs
    eta: int,
    early_stopping_rate: int = 0,
    fid_bounds: tuple[int, int] | tuple[float, float],
    # Extra
    seed: torch.Generator | None = None,
) -> dict[str, Any]:
    rung_to_fid, rung_sizes = brackets.calculate_sh_rungs(
        bounds=fid_bounds,
        eta=eta,
        early_stopping_rate=early_stopping_rate,
    )
    max_rung = max(rung_sizes)
    prior_dist = Prior.from_config(space.prior_config, space=space)

    # Below we will follow the "geomtric" spacing
    w_random = 1 / (1 + eta**rung_to_sample_for)
    w_prior = 1 - w_random

    completed: pd.DataFrame = table[table["perf"].notna()]  # type: ignore

    # To see if we activate incumbent sampling, we check:
    # 1) We have at least one fully complete run
    # 2) We have spent at least one full SH bracket worth of fidelity
    # 3) There is at least one rung with eta evaluations to get the top 1/eta configs of
    completed_rungs = completed.index.get_level_values("rung")
    one_complete_run_at_max_rung = (completed_rungs == max_rung).any()

    # For SH bracket cost, we include the fact we can continue runs,
    # i.e. resources for rung 2 discounts the cost of evaluating to rung 1,
    # only counting the difference in fidelity cost between rung 2 and rung 1.
    cost_per_rung = {i: rung_to_fid[i] - rung_to_fid.get(i - 1, 0) for i in rung_to_fid}

    cost_of_one_sh_bracket = sum(rung_sizes[r] * cost_per_rung[r] for r in rung_sizes)
    current_cost_used = sum(r * cost_per_rung[r] for r in completed_rungs)
    spent_one_sh_bracket_worth_of_fidelity = current_cost_used >= cost_of_one_sh_bracket

    # Check that there is at least rung with `eta` evaluations
    rung_counts = completed.groupby("rung").size()
    any_rung_with_eta_evals = (rung_counts == eta).any()

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
                config = prior_dist.sample_config(to=encoder)
            case "random":
                _sampler = Sampler.uniform(ndim=encoder.ndim)
                config = _sampler.sample_config(to=encoder)

        return config

    # Otherwise, we now further split the `prior` weight into `(prior, inc)`

    # 1. Select the top `1//eta` percent of configs at the highest rung that supports it
    rungs_with_at_least_eta = rung_counts[rung_counts >= eta].index  # type: ignore
    rung_table: pd.DataFrame = completed[  # type: ignore
        completed.index.get_level_values("rung") == rungs_with_at_least_eta.max()
    ]

    K = len(rung_table) // eta
    top_k_configs = rung_table.nsmallest(K, columns=["perf"])["config"].tolist()

    # 2. Get the global incumbent, and build a prior distribution around it
    inc = completed.loc[completed["perf"].idxmin()]["config"]
    inc_dist = Prior.from_config(inc, space=space)

    # 3. Calculate a ratio score of how likely each of the top K configs are under
    # the prior and inc distribution, weighing them by their position in the top K
    weights = torch.arange(K, 0, -1)
    top_k_pdf_inc = inc_dist.pdf_configs(top_k_configs, frm=encoder)
    top_k_pdf_prior = prior_dist.pdf_configs(top_k_configs, frm=encoder)

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
    policy = np.random.choice(["prior", "inc", "random"], p=[w_prior, w_inc, w_random])
    match policy:
        case "prior":
            return prior_dist.sample_config(to=encoder)
        case "random":
            _sampler = Sampler.uniform(ndim=encoder.ndim)
            return _sampler.sample_config(to=encoder)
        case "inc":
            assert inc is not None
            return mutate_config(
                inc,
                space=space,
                mutation_rate=inc_mutation_rate,
                std=inc_mutation_std,
                include_fidelity=False,
                seed=seed,
            )

    raise RuntimeError(f"Unknown policy: {policy}")
