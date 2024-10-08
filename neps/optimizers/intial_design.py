from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch

from neps.sampling import Sampler
from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.search_spaces.encoding import ConfigEncoder
    from neps.search_spaces.search_space import SearchSpace


def make_initial_design(
    space: SearchSpace,
    encoder: ConfigEncoder,
    sampler: Literal["sobol", "prior", "uniform"] | Sampler,
    sample_size: int | Literal["ndim"] | None = "ndim",
    sample_default_first: bool = True,
    sample_fidelity: (
        Literal["min", "max", True] | int | float | dict[str, int | float]
    ) = True,
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

            If "ndim", the number of configurations will be equal to the number of dimensions.
            If None, no configurations will be sampled.

        sample_default_first: Whether to sample the default configuration first.
        sample_fidelity:
            At what fidelity to sample the configurations, including the default.

            If set to "min" or "max", the configuration will be sampled
            at the minimum or maximum fidelity, respectively. If set to an integer
            or a float, the configuration will be sampled at that fidelity.
            When specified as a dictionary, the keys should be the names of the
            fidelity parameters and the values should be the target fidelities.
            If set to `True`, the configuration will have its fidelity randomly sampled.
        seed: The seed to use for the random number generation.

    """
    configs: list[dict[str, Any]] = []

    # First, we establish what fidelity to apply to them.
    match sample_fidelity:
        case "min":
            fids = {name: fid.lower for name, fid in space.fidelities.items()}
        case "max":
            fids = {name: fid.upper for name, fid in space.fidelities.items()}
        case True:
            fids = {name: hp.sample_value() for name, hp in space.fidelities.items()}
        case int() | float():
            if len(space.fidelities) != 1:
                raise ValueError(
                    "The target fidelity should be specified as a dictionary"
                    " if there are multiple fidelities or no fidelity should"
                    " be specified."
                    " Current search space has fidelities: "
                    f"{list(space.fidelities.keys())}"
                )
            name = next(iter(space.fidelities.keys()))
            fids = {name: sample_fidelity}
        case Mapping():
            missing_keys = set(space.fidelities.keys()) - set(sample_fidelity.keys())
            if any(missing_keys):
                raise ValueError(
                    f"Missing target fidelities for the following fidelities: "
                    f"{missing_keys}"
                )
            fids = sample_fidelity
        case _:
            raise ValueError(
                "Invalid value for `sample_default_at_target`. "
                "Expected 'min', 'max', True, int, float, or dict."
            )

    if sample_default_first:
        # TODO: No way to pass a seed to the sampler
        default = {
            name: hp.default if hp.default is not None else hp.sample_value()
            for name, hp in space.hyperparameters.items()
        }
        configs.append({**default, **fids})

    params = {**space.numerical, **space.categoricals}
    ndims = len(params)

    if sample_size == "ndim":
        sample_size = ndims
    elif sample_size is not None and not sample_size > 0:
        raise ValueError(
            "The sample size should be a positive integer if passing an int."
        )

    if sample_size is not None:
        match sampler:
            case "sobol":
                sampler = Sampler.sobol(ndim=len(params))
            case "uniform":
                sampler = Sampler.uniform(ndim=len(params))
            case "prior":
                sampler = Prior.from_parameters(params.values())
            case _:
                sampler = sampler

        encoded_configs = sampler.sample(sample_size * 2, to=encoder.domains, seed=seed)
        uniq_x = torch.unique(encoded_configs, dim=0)
        sample_configs = encoder.decode(uniq_x[:sample_size])
        configs.extend([{**config, **fids} for config in sample_configs])

    return configs
