"""Priors for search spaces.

Loosely speaking, they are joint distributions over multiple independent
variables, i.e. each column of a tensor is assumed to be independent and
can be acted on independently.

See the class doc description of [`Prior`][neps.priors.Prior] for more details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Container, Mapping, Protocol
from typing_extensions import override

import torch

from neps.distributions import DistributionOverDomain, TruncatedNormal
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain

if TYPE_CHECKING:
    from torch.distributions import Distribution


class Prior(Sampler, Protocol):
    """A protocol for priors over search spaces.

    Extends from the [`Sampler`][neps.samplers.Sampler] protocol.

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

    def log_prob(
        self,
        x: torch.Tensor,
        *,
        frm: list[Domain] | Domain,
    ) -> torch.Tensor:
        """Compute the log probability of values in `x` under a prior.

        The last dimenion of `x` is assumed to be independent, such that the
        log probability of the entire tensor is the sum of the log
        probabilities of each element in that dimension.

        For example, if `x` is of shape `(n_samples, n_dims)`, then the
        you will be given back a tensor of shape `(n_samples,)` with the
        each entry being the log probability of the corresponding sample.

        Args:
            x: Tensor of shape (..., n_dims)
                In the case of a 1D tensor, the shape is assumed to be (n_dims,)
            frm: The domain of the values in `x`. If a single domain, then all the
                values are assumed to be from that domain, otherwise each column
                `n_dims` in (n_samples, n_dims) is from the corresponding domain.

        Returns:
            Tensor of shape (...,), with the last dimension reduced out. In the
            case that only single dimensional tensor is passed, the returns value
            is a scalar.
        """
        ...

    def prob(self, x: torch.Tensor, *, frm: Domain | list[Domain]) -> torch.Tensor:
        """Compute the probability of values in `x` under a prior.

        See [`log_prob()`][neps.priors.Prior.log_prob] for details on shapes.
        """
        return torch.exp(self.log_prob(x, frm=frm))

    @classmethod
    def uniform(cls, ncols: int) -> UniformPrior:
        """Create a uniform prior for a given list of domains.

        Args:
            ncols: The number of columns in the tensor to sample.
        """
        return UniformPrior(ncols=ncols)

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

        return CenteredPrior(distributions=distributions)


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

    distributions: list[DistributionOverDomain]
    """Distributions along with the corresponding domains they sample from."""

    _distribution_domains: list[Domain] = field(init=False, repr=False)

    def __post_init__(self):
        self._distribution_domains = [dist.domain for dist in self.distributions]

    @property
    @override
    def ncols(self) -> int:
        return len(self.distributions)

    @override
    def log_prob(self, x: torch.Tensor, *, frm: list[Domain] | Domain) -> torch.Tensor:
        if x.ndim == 0:
            raise ValueError("Expected a tensor of shape (..., ncols).")

        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Cast all values from the value domains to the domain of the sampler.
        sample_domain_tensor = Domain.translate(
            x,
            frm=frm,
            to=self._distribution_domains,
        )

        # Calculate the log probabilities of the sample domain tensors under their
        # respective distributions.
        log_probs = torch.cat(
            [
                dist.distribution.log_prob(sample_domain_tensor[:, i])
                for i, dist in enumerate(self.distributions)
            ],
            dim=-1,
        )
        return torch.sum(log_probs, dim=-1)

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain],
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        _out_shape = (
            torch.Size((n, self.ncols))
            if isinstance(n, int)
            else torch.Size((*n, self.ncols))
        )
        _n = torch.Size((n,)) if isinstance(n, int) else n

        out = torch.empty(_out_shape, device=device, dtype=torch.float64)
        for i, dist in enumerate(self.distributions):
            out[..., i] = dist.distribution.sample(_n)

        return Domain.translate(out, frm=self._distribution_domains, to=to)


@dataclass
class UniformPrior(Prior):
    """A prior that is uniform over a given domain.

    Uses a UnitUniform under the hood before converting to the value domain.
    """

    ncols: int
    """The number of columns in the tensor to sample from."""

    _unit_uniform: Distribution = field(init=False, repr=False)

    def __post_init__(self):
        self._unit_uniform = torch.distributions.Uniform(0.0, 1.0)

    @override
    def log_prob(self, x: torch.Tensor, *, frm: Domain | list[Domain]) -> torch.Tensor:
        sample_domain_tensor = Domain.translate(x, frm=frm, to=UNIT_FLOAT_DOMAIN)
        return torch.sum(self._unit_uniform.log_prob(sample_domain_tensor), dim=-1)

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain],
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        _n = (
            torch.Size((n, self.ncols))
            if isinstance(n, int)
            else torch.Size((*n, self.ncols))
        )
        samples = torch.rand(_n, device=device, dtype=torch.float64)
        return Domain.translate(samples, frm=UNIT_FLOAT_DOMAIN, to=to)


@dataclass
class WeightedPrior(Prior):
    """A prior consisting of multiple priors with weights."""

    priors: list[Prior]
    weights: torch.Tensor
    probabilities: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.priors) < 2:
            raise ValueError(f"At least two priors must be given. Got {len(self.priors)}")

        if self.weights.ndim != 1:
            raise ValueError("Weights must be a 1D tensor.")

        if len(self.priors) != len(self.weights):
            raise ValueError("The number of priors and weights must be the same.")

        self.probabilities = self.weights / self.weights.sum()

    @override
    def log_prob(self, x: torch.Tensor, *, frm: Domain | list[Domain]) -> torch.Tensor:
        # OPTIM: Avoid an initial allocation by using the output of the first
        # distribution to store the weighted probabilities
        itr = zip(self.probabilities, self.priors)
        first_prob, first_prior = next(itr)

        weighted_probs = first_prob * first_prior.log_prob(x, frm=frm)
        for prob, prior in itr:
            weighted_probs += prob * prior.log_prob(x, frm=frm)

        return weighted_probs

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain],
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        # Calculate the total number of samples required
        if isinstance(n, int):
            total_samples = n
            output_shape = (n, self.ncols)
        else:
            total_samples = reduce(lambda x, y: x * y, n)
            output_shape = (*n, self.ncols)

        # Randomly select which prior to sample from for each of the total_samples
        chosen_priors = torch.empty((total_samples,), device=device, dtype=torch.int64)
        chosen_priors = torch.multinomial(
            self.probabilities,
            total_samples,
            replacement=True,
            out=chosen_priors,
        )

        # Create an empty tensor to hold all samples
        output_samples = torch.empty(
            (total_samples, self.ncols), device=device, dtype=torch.float64
        )

        # Loop through each prior and its associated indices
        for i, prior in enumerate(self.priors):
            # Find indices where the chosen prior is i
            _i = torch.tensor(i, dtype=torch.int64, device=device)
            indices = torch.where(chosen_priors == _i)[0]

            if len(indices) > 0:
                # Sample from the prior for the required number of indices
                samples_from_prior = prior.sample(len(indices), to=to, device=device)
                output_samples[indices] = samples_from_prior

        # Reshape to the output shape including ncols dimension
        output_samples = output_samples.view(output_shape)

        return Domain.translate(output_samples, frm=UNIT_FLOAT_DOMAIN, to=to)
