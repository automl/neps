from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neps.sampling import Uniform
from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.space import ConfigEncoder
    from neps.space.parameters import Parameter


@dataclass
class MOPriorSampler:
    """A Sampler implementing a sampling strategy for multi-objective
    optimization with priors."""

    prior_dists: Mapping[str, Prior]

    parameters: Mapping[str, Parameter]

    encoder: ConfigEncoder

    sampler_type: Literal["mopriors", "mix_random", "etaprior"] = "mopriors"

    eta: int | None = None

    def __post_init__(self) -> None:
        if self.sampler_type == "etaprior":
            assert self.eta is not None, "Eta must be set for `sampler_type='etaprior'`"

    @classmethod
    def dists_from_centers_and_confidences(
        cls,
        parameters: Mapping[str, Parameter],
        prior_centers: Mapping[str, Mapping[str, float]],
        confidence_values: Mapping[str, Mapping[str, float]],
    ) -> Mapping[str, Prior]:
        """Creates a mapping of prior distributions from the given centers and
        confidence values.
        Args:
            parameters: The parameters to sample from.
            prior_centers: The priors to use for sampling.
            confidence_values: The confidence values for the priors.
        Returns:
            A mapping of prior distributions.
        """
        _priors = {}

        for key, _prior_center in prior_centers.items():
            assert isinstance(_prior_center, dict), (
                f"Expected prior center values to be a dict, got {type(_prior_center)}"
            )
            assert key in confidence_values, (
                f"Expected confidence values to contain {key}, "
                f"got {confidence_values.keys()}"
            )
            _priors[key] = Prior.from_parameters(
                parameters=parameters,
                center_values=_prior_center,
                confidence_values=confidence_values[key],
            )
        return _priors

    @classmethod
    def create_sampler(
        cls,
        parameters: Mapping[str, Parameter],
        prior_centers: Mapping[str, Mapping[str, float]],
        confidence_values: Mapping[str, Mapping[str, float]],
        encoder: ConfigEncoder,
        *,
        sampler_type: Literal["mopriors", "mix_random", "etaprior"] = "mopriors",
        eta: int | None = None,
    ) -> MOPriorSampler:
        """Creates a MOPriorSampler instance.

        Args:
            parameters: The parameters to sample from.
            prior_centers: The priors to use for sampling.
            confidence_values: The confidence values for the priors.
            encoder: The encoder to use for encoding and decoding configurations
                into tensors.
            sampler_type: The type of sampler to create. Defaults to "mopriors".
            eta: The eta value to use for the SH bracket. Only used if
                `sampler_type` is "etaprior".

        Returns:
            The MOPriorSampler instance.
        """
        _priors = cls.dists_from_centers_and_confidences(
            parameters=parameters,
            prior_centers=prior_centers,
            confidence_values=confidence_values,
        )

        return MOPriorSampler(
            prior_dists=_priors,
            parameters=parameters,
            encoder=encoder,
            sampler_type=sampler_type,
            eta=eta,
        )

    def sample_config(self) -> dict[str, Any]:
        """Samples a configuration using the MOPriors algorithm.

        Returns:
            The sampled configuration.
        """
        match self.sampler_type:
            case "mopriors":
                # If using MOPriors, sample from the priors
                _sampler = "prior"
            case "mix_random":
                _sampler = np.random.choice(
                    ["random", "prior"],
                )
            case "etaprior":
                # If using EtaPrior, sample from the priors by 1/eta probability
                assert self.eta is not None, (
                    "Eta must be set for `sampler_type='etaprior'`"
                )
                _sampler = np.random.choice(
                    ["prior", "random"],
                    p=[1 / self.eta, 1 - (1 / self.eta)],
                )

        match _sampler:
            case "random":
                # If mixing random sampling, sample a random configuration
                return Uniform(ndim=len(self.parameters)).sample_config(to=self.encoder)

            case "prior":
                # If not mixing random sampling, sample from the priors
                # Randomly choose one of the prior distributions
                _prior_choice: Prior = np.random.choice(
                    list(self.prior_dists.values()),
                )

                # Sample a configuration from the chosen prior
                return _prior_choice.sample_config(to=self.encoder)

            case _:
                raise ValueError("Shouldn't reach here!")
