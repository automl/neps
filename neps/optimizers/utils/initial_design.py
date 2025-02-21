from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch

from neps.optimizers.priorband import mutate_config
from neps.sampling import Prior, Sampler
from neps.space import Grammar
from neps.space.grammar import RandomSampler, MutationSampler, GrammarSampler

if TYPE_CHECKING:
    from neps.space import ConfigEncoder, Parameter


def make_initial_design(
    *,
    parameters: Mapping[str, Parameter | Grammar],
    encoder: ConfigEncoder,
    sampler: Literal["sobol", "prior", "uniform"] | Sampler,
    sample_size: int | Literal["ndim"] | None = "ndim",
    sample_prior_first: bool = True,
    grammar_mutant_selector:(
        tuple[Literal["symbol"], str]
        | tuple[Literal["depth"], int | range]
        | tuple[Literal["climb"], int | range]
    ) = ("climb", range(1, 4)),
    grammar_max_mutation_depth: int = 3,
    seed: torch.Generator | None = None,
) -> list[dict[str, Any]]:
    """Generate the initial design of the optimization process.

    Args:
        space: The search space to use.
        encoder: The encoder to use for encoding/decoding configurations.
        sampler: The sampler to use for the initial design.

            If set to "sobol", a Sobol sequence will be used.
            If set to "uniform", a uniform random sampler will be used.
            If set to "prior", a prior sampler will be used, based on the defaults,
                and confidence scores of the hyperparameters.
            If set to a custom sampler, the sampler will be used directly.

        sample_size:
            The number of configurations to sample.

            If "ndim", the number of configs will be equal to the number of dimensions.
            If None, no configurations will be sampled.

        sample_prior_first: Whether to sample the prior configuration first.
        grammar_mutant_selector: Please see [`select()`][neps.space.grammar.select].
        grammar_max_mutation_depth: How deep to enumerate mutants of a prior for the
            grammar.
        seed: The seed to use for the random number generation.

    """
    configs: list[dict[str, Any]] = []
    numerics = {k: p for k, p in parameters.items() if not isinstance(p, Grammar)}
    grammars = {k: p for k, p in parameters.items() if isinstance(p, Grammar)}
    if sample_prior_first:
        grammar_priors: dict[str, str] = {
            k: (
                g.prior
                if g.prior is not None
                # Ew sorry
                else RandomSampler(g).sample(1)[0].to_string()
            )
            for k, g in grammars.items()
        }
        numeric_priors: dict[str, Any] = {
            name: p.prior if p.prior is not None else p.center
            for name, p in numerics.items()
        }
        configs.append({**numeric_priors, **grammar_priors})

    numeric_ndims = len(numerics)
    grammar_expansion_count = sum(g._expansion_count for g in grammars.values())
    if sample_size == "ndim":
        # TODO: Not sure how to handle graphs here properly here to be honest
        sample_size = numeric_ndims + grammar_expansion_count
    elif sample_size is not None and not sample_size > 0:
        raise ValueError(
            "The sample size should be a positive integer if passing an int."
        )

    if sample_size is not None:
        # Numeric sampling
        match sampler:
            case "sobol":
                numeric_sampler = Sampler.sobol(ndim=numeric_ndims)
                grammar_sampler = GrammarSampler.random(grammars)
            case "uniform":
                numeric_sampler = Sampler.uniform(ndim=numeric_ndims)
                grammar_sampler = GrammarSampler.random(grammars)
            case "prior":
                numeric_sampler = Prior.from_parameters(numerics)
                grammar_sampler = GrammarSampler.prior(
                    grammars,
                    mutant_selector=grammar_mutant_selector,
                    max_mutation_depth=grammar_max_mutation_depth
                )
            case _:
                pass

        # TODO: Replace with something more solid
        # Grammar sampling
        for k, g in grammars.items():

        encoded_configs = sampler.sample(sample_size * 2, to=encoder.domains, seed=seed)
        uniq_x = torch.unique(encoded_configs, dim=0)
        sample_configs = encoder.decode(uniq_x[:sample_size])
        configs.extend(sample_configs)

    return configs
