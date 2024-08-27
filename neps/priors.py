"""Priors for search spaces.

Loosely speaking, they are joint distributions over multiple independent
variables, i.e. each column of a tensor is assumed to be independent and
can be acted on independently.

They are not a `torch.distributions.Distribution` subclass as methods like
`entropy` and `kl_divergence` are just more difficult to implement
(not impossible, just more difficult and not needed right now).

See the class doc description of [`Prior`][neps.priors.Prior] for more details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Container, Mapping, Protocol
from typing_extensions import override

import torch

from neps.distributions import DistributionOverDomain, TruncatedNormal
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain

if TYPE_CHECKING:
    from torch.distributions import Distribution


class Prior(Protocol):
    """A protocol for priors over search spaces.

    At it's core, the two methods that need to be implemented are
    `log_prob` and `sample`. The `log_prob` method should return the
    log probability of a given tensor of samples under its distribution.
    The `sample` method should return a tensor of samples from distribution.

    All values given to the `log_prob` and the ones returned from the
    `sample` method are assumed to be in the value domain of the prior,
    i.e. the [`.domains`][neps.priors.Prior] attribute.

    !!! warning

        The domain in which samples are actually drawn from not necessarily
        need to match that of the value domain. For example, the
        [`UniformPrior`][neps.priors.UniformPrior] class uses a unit uniform
        distribution to sample from the unit interval before converting
        samples to the value domain.

        **As a result, the `log_prob` and `prob` method may not give the same
        values as you might expect for a distribution over the value domain.**

        For example, consider a value domain `[0, 1e9]`. You might expect
        the `pdf` to be `1e-9` (1 / 1e9) for any given value inside the domain.
        However, since the `UniformPrior` samples from the unit interval, the `pdf` will
        actually be `1` (1 / 1) for any value inside the domain.
    """

    domains: list[Domain]
    """Domain of values which this prior acts upon.

    Each domain corresponds to the corresponding `ndim` in a tensor
    (n_samples, ndim).
    """

    device: torch.device | None
    """Device to place the tensors on."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of values in `x` under a prior.

        All columns of `x` are assumed to be independent, such that the
        log probability of the entire tensor is the sum of the log
        probabilities of each column.

        Args:
            x: Tensor of shape (n_samples, n_dims)
                In the case of a 1D tensor, the shape is assumed to be (n_dims,)

        Returns:
            Tensor of shape (n_samples,) with the log probabilities of each. In the
            case that only single dimensional tensor is passed, the returns value
            is a scalar.
        """
        ...

    def sample(self, n: int) -> torch.Tensor:
        """Sample from the prior.

        Args:
            n: Number of samples to draw.

        Returns:
            Tensor of shape (n, n_dims) with the samples.
        """
        ...

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the probability of values in `x` under a prior.

        See [`log_prob()`][neps.priors.Prior.log_prob] for details on shapes.
        """
        return torch.exp(self.log_prob(x))

    @classmethod
    def uniform(
        cls,
        domains: Mapping[str, Domain] | list[Domain],
        *,
        device: torch.device | None = None,
    ) -> UniformPrior:
        """Create a uniform prior for a given list of domains.

        Args:
            domains: domains over which to have a uniform prior.
            device: Device to place the tensors on.
        """
        domains = domains if isinstance(domains, list) else list(domains.values())
        return UniformPrior(domains=domains, device=device)

    @classmethod
    def make_centered(
        cls,
        domains: Mapping[str, Domain],
        centers: Mapping[str, tuple[Any, float]],
        *,
        categoricals: Container[str] = (),
        device: torch.device | None = None,
    ) -> CenteredPrior:
        """Create a prior for a given list of domains.

        Will use a `TruncatedNormal` distribution for all parameters,
        except those contained within `categoricals`, which will
        use a `Categorical` instead. If no center is given for a domain,
        a uniform prior will be used.

        For non-categoricals, this will be interpreted as the mean and
        std `(1 - confidence)` for a truncnorm. For categorical values,
        the _center_ will contain a probability mass of `confidence` with
        the remaining `(1 - confidence)` probability mass distributed uniformly
        amongest the other choices.

        The order of the items in `domains` matters and should align
        with any tensors that you will use to evaluate from the prior.
        I.e. the first domain in `domains` will be the first column
        of a tensor that this prior can be used on.

        Args:
            domains: domains over which to have a centered prior.
            centers: centers for the priors. Should be a mapping
                from the domain name to the center value and confidence level.
                If no center is given, a uniform prior will be used.

                !!! warning

                    The values contained in centers should be contained within the
                    domain. All confidence levels should be within the `[0, 1]` range.

            categoricals: The names of the domains that are categorical and which
                a `Categorical` distribution will be used, rather than a
                `TruncatedNormal`.

                !!! warning

                    Categoricals require that the corresponding domain has a
                    `.cardinality`, i.e. it is not a float/continuous domain.

            device: Device to place the tensors on.


        Returns:
            A prior for the search space.
        """
        for name, (_, confidence) in centers.items():
            if not 0 <= confidence <= 1:
                raise ValueError(
                    f"Confidence level for {name} must be in the range [0, 1]."
                    f" Got {confidence}."
                )

        for name in domains:
            if name not in centers:
                raise ValueError(
                    f"Center for {name} is missing. "
                    f"Please provide a center for all domains."
                )

        distributions: list[DistributionOverDomain] = []
        for name, domain in domains.items():
            center_confidence = centers.get(name)
            if center_confidence is None:
                dist = DistributionOverDomain(
                    distribution=torch.distributions.Uniform(0.0, 1.0),
                    domain=UNIT_FLOAT_DOMAIN,
                )
                continue

            center, confidence = center_confidence
            if name in categoricals:
                if domain.cardinality is None:
                    raise ValueError(
                        f"{name} is not a finite domain and cannot be used as a"
                        " categorical. Please remove it from the categoricals list."
                    )

                if not isinstance(center, int):
                    raise ValueError(
                        f"{name} is a categorical domain and should have an integer"
                        f" center. Got {center} of type {type(center)}."
                    )

                remaining_weight = 1 - confidence
                distributed_weight = remaining_weight / (domain.cardinality - 1)
                weights = torch.full(
                    (domain.cardinality,),
                    distributed_weight,
                    device=device,
                    dtype=torch.float64,
                )

                weights[center] = confidence

                dist = DistributionOverDomain(
                    distribution=torch.distributions.Categorical(probs=weights),
                    domain=domain,
                )
                distributions.append(dist)
                continue

            # We place a truncnorm over a unitnorm
            unit_center = domain.to_unit(
                torch.tensor(center, device=device, dtype=torch.float64)
            )
            dist = DistributionOverDomain(
                distribution=TruncatedNormal(
                    loc=unit_center,
                    scale=(1 - confidence),
                    a=0.0,
                    b=1.0,
                    device=device,
                ),
                domain=UNIT_FLOAT_DOMAIN,
            )
            distributions.append(dist)

        return CenteredPrior(
            domains=list(domains.values()), distributions=distributions, device=device
        )


@dataclass
class CenteredPrior(Prior):
    """A prior that is centered around a given value with a given confidence.

    This prior is useful for creating priors for search spaces where the
    values are centered around a given value with a given confidence level.

    You can use a `torch.distribution.Uniform` for any values which do
    not have a center and confidence level, i.e. no prior information.

    You can create this class more easily using
    [`Prior.make_centered()`][neps.priors.Prior.make_centered].
    """

    domains: list[Domain]
    """Domain of values."""

    device: torch.device | None
    """Device to place the tensors on."""

    distributions: list[DistributionOverDomain]
    """Distributions along with the corresponding domains they sample from."""

    _distribution_domains: list[Domain] = field(init=False, repr=False)

    def __post_init__(self):
        self._distribution_domains = [dist.domain for dist in self.distributions]

    @override
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # Cast all values from the value domains to the domain of the sampler.
        sample_domain_tensor = Domain.cast_many(
            x, frm=self.domains, to=self._distribution_domains
        )

        # Calculate the log probabilities of the sample domain tensors under their
        # respective distributions.
        log_probs = torch.cat(
            [
                dist.distribution.log_prob(sample_domain_tensor[:, i])
                for i, dist in enumerate(self.distributions)
            ],
            dim=1,
        )
        return torch.sum(log_probs, dim=1)

    @override
    def sample(self, n: int) -> torch.Tensor:
        buffer = torch.empty(
            n,
            len(self.distributions),
            device=self.device,
            dtype=torch.float64,
        )

        size = torch.Size((n,))
        for i, (value_domain, frm) in enumerate(zip(self.domains, self.distributions)):
            samples = frm.distribution.sample(size)
            buffer[:, i] = value_domain.cast(samples, frm=frm.domain)

        return buffer


@dataclass
class UniformPrior(Prior):
    """A prior that is uniform over a given domain.

    Uses a UnitUniform under the hood before converting to the value domain.
    """

    domains: list[Domain]
    """Domain of values."""

    device: torch.device | None
    """Device to place the tensors on."""

    _unit_uniform: Distribution = field(init=False, repr=False)

    def __post_init__(self):
        self._unit_uniform = torch.distributions.Uniform(0.0, 1.0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of values in `x` under a prior.

        All columns of `x` are assumed to be independent, such that the
        log probability of the entire tensor is the sum of the log
        probabilities of each column.

        Args:
            x: Tensor of shape (n_samples, n_dims)
                In the case of a 1D tensor, the shape is assumed to be (n_dims,)

        Returns:
            Tensor of shape (n_samples,) with the log probabilities of each. In the
            case that only single dimensional tensor is passed, the returns value
            is a scalar.
        """
        sample_domain_tensor = Domain.cast_many(x, frm=self.domains, to=UNIT_FLOAT_DOMAIN)
        return torch.sum(self._unit_uniform.log_prob(sample_domain_tensor), dim=1)

    def sample(self, n: int) -> torch.Tensor:
        """Sample from the prior.

        Args:
            n: Number of samples to draw.

        Returns:
            Tensor of shape (n, n_dims) with the samples.
        """
        samples = torch.rand(
            n,
            len(self.domains),
            device=self.device,
            dtype=torch.float64,
        )
        return Domain.cast_many(samples, frm=UNIT_FLOAT_DOMAIN, to=self.domains)
