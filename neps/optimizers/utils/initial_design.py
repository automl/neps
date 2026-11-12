from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch

from neps.sampling import Prior, Sampler

if TYPE_CHECKING:
    from neps.space import ConfigEncoder, SearchSpace
    from neps.space.neps_spaces.parameters import PipelineSpace
    from neps.space.parameters import Parameter


def make_initial_design(
    *,
    parameters: Mapping[str, Parameter],
    encoder: ConfigEncoder,
    sampler: Literal["sobol", "prior", "uniform"] | Sampler,
    sample_size: int | Literal["ndim"] | None = "ndim",
    sample_prior_first: bool = True,
    seed: torch.Generator | None = None,
) -> list[dict[str, Any]]:
    """Generate the initial design of the optimization process.

    Args:
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
        seed: The seed to use for the random number generation.

    """
    configs: list[dict[str, Any]] = []
    if sample_prior_first:
        configs.append(
            {
                name: p.prior if p.prior is not None else p.center
                for name, p in parameters.items()
            }
        )

    ndims = len(parameters)
    if sample_size == "ndim":
        sample_size = ndims
    elif sample_size is not None and not sample_size > 0:
        raise ValueError(
            "The sample size should be a positive integer if passing an int."
        )

    if sample_size is not None:
        match sampler:
            case "sobol":
                sampler = Sampler.sobol(ndim=ndims)
            case "uniform":
                sampler = Sampler.uniform(ndim=ndims)
            case "prior":
                sampler = Prior.from_parameters(parameters)
            case _:
                pass

        encoded_configs = sampler.sample(sample_size * 2, to=encoder.domains, seed=seed)
        uniq_x = torch.unique(encoded_configs, dim=0)
        sample_configs = encoder.decode(uniq_x[:sample_size])
        configs.extend(sample_configs)

    return configs


def complete_initial_design(
    initial_design: Sequence[Mapping[str, Any]],
    space: SearchSpace,
    *,
    parameters: Mapping[str, Parameter],
) -> list[dict[str, Any]]:
    """Validate an initial design given from outside the optimizer and complete it.

    Each config must set every parameter in `parameters` (the parameters the
    optimizer searches over). Constants of the space are always set, and fidelities
    not given in a config are set to their maximum.

    Args:
        initial_design: The configurations to evaluate first, in order.
        space: The (classic) search space the optimizer works on.
        parameters: The parameters the optimizer searches over.

    Returns:
        The completed configurations, in the same order.
    """
    known = set(space.elements)
    completed: list[dict[str, Any]] = []
    for i, config in enumerate(initial_design):
        if missing := set(parameters) - set(config):
            raise ValueError(
                f"Initial design config {i} misses parameters {sorted(missing)}: {config}"
            )
        if unknown := set(config) - known:
            raise ValueError(
                f"Initial design config {i} has unknown parameters {sorted(unknown)}:"
                f" {config}"
            )
        completed.append(
            {
                **{name: f.upper for name, f in space.fidelities.items()},
                **config,
                **space.constants,
            }
        )
    return completed


def next_from_initial_design(
    initial_design: Sequence[Mapping[str, Any]] | None,
    trials: Mapping[str, Any],
    n: int,
) -> list[dict[str, Any]]:
    """Return the next (at most `n`) configs of a given initial design.

    The position in the design is the number of trials NePS knows about (pending,
    failed and successful), so every optimizer given the same design proposes the
    same configs first, and multiple workers do not propose the same config twice.
    """
    if not initial_design:
        return []
    start = len(trials)
    return [dict(config) for config in initial_design[start : start + n]]


def sample_initial_design(
    pipeline_space: SearchSpace | PipelineSpace | Mapping[str, Any],
    size: int,
    *,
    seed: int,
    sampler: Literal["sobol", "uniform"] = "uniform",
    sample_fidelity: bool = False,
) -> list[dict[str, Any]]:
    """Sample an initial design to give to several optimizers so they start the same.

    Pass the result as `initial_design=` to e.g.
    [`random_search`][neps.optimizers.algorithms.random_search],
    [`bayesian_optimization`][neps.optimizers.algorithms.bayesian_optimization] or
    [`optuna_motpe`][neps.optimizers.algorithms.optuna_motpe].

    Args:
        pipeline_space: The search space to sample from.
        size: The number of configurations.
        seed: The seed; the same seed gives the same design.
        sampler: "uniform" (random) or "sobol" (scrambled Sobol sequence).
        sample_fidelity: Whether to sample fidelities like any other parameter
            (`ignore_fidelity=True` in the optimizers). If False they are set to
            their maximum.
    """
    from neps.space import ConfigEncoder
    from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
    from neps.space.neps_spaces.parameters import PipelineSpace
    from neps.space.parsing import convert_to_space

    space = convert_to_space(pipeline_space)
    if isinstance(space, PipelineSpace):
        converted = convert_neps_to_classic_search_space(space)
        if converted is None:
            raise ValueError("initial designs are only supported for HPO search spaces.")
        space = converted
    parameters = (
        {**space.searchables, **space.fidelities}
        if sample_fidelity
        else {**space.searchables}
    )
    encoder = ConfigEncoder.from_parameters(parameters)
    match sampler:
        case "sobol":
            _sampler = Sampler.sobol(ndim=len(parameters))
        case "uniform":
            _sampler = Sampler.uniform(ndim=len(parameters))
        case _:
            raise ValueError(f"Unknown sampler {sampler!r}")
    # The samplers do not all support a seed argument, so seed a forked torch RNG
    # (this does not change the global RNG state).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoded = _sampler.sample(size * 4, to=encoder.domains)
    # Drop duplicates (possible with integers/categoricals) keeping the sample order
    design: list[dict[str, Any]] = []
    for config in encoder.decode(encoded):
        if config not in design:
            design.append(config)
        if len(design) == size:
            break
    if len(design) < size:
        raise ValueError(
            f"Could only sample {len(design)} unique configs, but {size} were requested."
        )
    return complete_initial_design(design, space, parameters=parameters)
