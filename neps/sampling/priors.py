"""Priors for search spaces.

Loosely speaking, they are joint distributions over multiple independent
variables, i.e. each column of a tensor is assumed to be independent and
can be acted on independently.

See the class doc description of [`Prior`][neps.sampling.Prior] for more details.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch

from neps.sampling.distributions import (
    UNIT_UNIFORM_DIST,
    TorchDistributionWithDomain,
    TruncatedNormal,
)
from neps.sampling.samplers import Sampler
from neps.space import Categorical, ConfigEncoder, Domain, Float, Integer

if TYPE_CHECKING:
    from torch.distributions import Distribution


class Prior(Sampler):
    """A protocol for priors over search spaces.

    Extends from the [`Sampler`][neps.sampling.Sampler] protocol.

    At it's core, the two methods that need to be implemented are
    `log_pdf` and `sample`. The `log_pdf` method should return the
    log probability of a given tensor of samples under its distribution.
    The `sample` method should return a tensor of samples from distribution.

    All values given to the `log_pdf` and the ones returned from the
    `sample` method are assumed to be in the value domain of the prior,
    i.e. the [`.domains`][neps.sampling.Prior] attribute.

    !!! warning

        The domain in which samples are actually drawn from not necessarily
        need to match that of the value domain. For example, the
        [`Uniform`][neps.sampling.Uniform] class uses a unit uniform
        distribution to sample from the unit interval before converting
        samples to the value domain.

        **As a result, the `log_pdf` and `pdf` method may not give the same
        values as you might expect for a distribution over the value domain.**

        For example, consider a value domain `[0, 1e9]`. You might expect
        the `pdf` to be `1e-9` (1 / 1e9) for any given value inside the domain.
        However, since the `Uniform` samples from the unit interval, the `pdf` will
        actually be `1` (1 / 1) for any value inside the domain.
    """

    @abstractmethod
    def log_pdf(
        self,
        x: torch.Tensor,
        *,
        frm: ConfigEncoder | list[Domain] | Domain,
    ) -> torch.Tensor:
        """Compute the log pdf of values in `x` under a prior.

        The last dimenion of `x` is assumed to be independent, such that the
        log pdf of the entire tensor is the sum of the log
        pdf of each element in that dimension.

        For example, if `x` is of shape `(n_samples, n_dims)`, then the
        you will be given back a tensor of shape `(n_samples,)` with the
        each entry being the log pdf of the corresponding sample.

        Args:
            x: Tensor of shape (..., n_dims)
                In the case of a 1D tensor, the shape is assumed to be (n_dims,)
            frm: The domain of the values in `x`. If a single domain, then all the
                values are assumed to be from that domain, otherwise each column
                `n_dims` in (n_samples, n_dims) is from the corresponding domain.
                If a `ConfigEncoder` is passed in, it will just take it's domains
                for use.

        Returns:
            Tensor of shape (...,), with the last dimension reduced out. In the
            case that only single dimensional tensor is passed, the returns value
            is a scalar.
        """

    def pdf(
        self, x: torch.Tensor, *, frm: ConfigEncoder | Domain | list[Domain]
    ) -> torch.Tensor:
        """Compute the pdf of values in `x` under a prior.

        See [`log_pdf()`][neps.sampling.Prior.log_pdf] for details on shapes.
        """
        return torch.exp(self.log_pdf(x, frm=frm))

    def pdf_configs(self, x: list[dict[str, Any]], *, frm: ConfigEncoder) -> torch.Tensor:
        """Compute the pdf of values in `x` under a prior.

        See [`log_pdf()`][neps.sampling.Prior.log_pdf] for details on shapes.
        """
        return self.pdf(frm.encode(x), frm=frm)

    @classmethod
    def uniform(cls, ncols: int) -> Uniform:
        """Create a uniform prior for a given list of domains.

        Args:
            ncols: The number of columns in the tensor to sample.
        """
        return Uniform(ndim=ncols)

    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, Categorical | Float | Integer],
        *,
        center_values: Mapping[str, Any] | None = None,
        confidence_values: Mapping[str, float] | None = None,
    ) -> CenteredPrior:
        """Create a prior distribution from dict of parameters.

        Args:
            parameters: The parameters to createa a prior from. Will look
                at the `.prior` and `.prior_confidence` of the parameters
                to create a truncated normal.

                Any parameters that do not have a `.prior` will be covered by
                a uniform distribution.
            center_values: Any values that should be used instead of the
                parameter's `.prior`.
            confidence_values: Any additional values that should be
                used for determining the strength of the prior. Values should
                be between 0 and 1. Overwrites whatever is set by default in
                the `.prior-confidence`.

        Returns:
            The prior distribution
        """
        _mapping = {"low": 0.25, "medium": 0.5, "high": 0.75}

        center_values = center_values or {}
        confidence_values = confidence_values or {}
        domains: list[Domain] = []
        centers: list[tuple[Any, float] | None] = []

        for name, hp in parameters.items():
            domains.append(hp.domain)

            default = center_values.get(name, hp.prior)
            if default is None:
                centers.append(None)
                continue

            confidence_score = confidence_values.get(name, _mapping[hp.prior_confidence])
            center = hp.choices.index(default) if isinstance(hp, Categorical) else default
            centers.append((center, confidence_score))

        return Prior.from_domains_and_centers(domains=domains, centers=centers)

    @classmethod
    def from_domains_and_centers(
        cls,
        domains: Iterable[Domain] | ConfigEncoder,
        centers: Iterable[None | tuple[int | float, float]],
        *,
        device: torch.device | None = None,
    ) -> CenteredPrior:
        """Create a prior for a given list of domains.

        This is a lower level version of
        [`from_parameters()`][neps.sampling.Prior.from_parameters] which
        requires a full specification of the domains and the centers.

        Will use a `TruncatedNormal` distribution for all parameters,
        except those who have a domain marked with `is_categorical=True`,
        using a `Categorical` distribution instead.
        If the center for a given domain is `None`, a uniform prior
        will be used instead.

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
            centers: centers for the priors, i.e. the mode of the prior for that
                domain, along with the confidence of that mode, which get's
                re-interpreted as the std of the truncnorm or the probability
                mass for the categorical.

                If `None`, a uniform prior will be used.

                !!! warning

                    The values contained in centers should be contained within the
                    domain. All confidence levels should be within the `[0, 1]` range.

            device: Device to place the tensors on for distributions.

        Returns:
            A prior for the search space.
        """
        match domains:
            case ConfigEncoder():
                domains = domains.domains
            case _:
                domains = list(domains)

        distributions: list[TorchDistributionWithDomain] = []
        for domain, center_conf in zip(domains, centers, strict=True):
            # If the center is None, we use a uniform distribution. We try to match
            # the distributions to all be unit uniform as it can speed up sampling when
            # consistentaly the same. This still works for categoricals
            if center_conf is None:
                if domain.is_categorical:
                    # Uniform categorical
                    n_cats = domain.cardinality
                    assert n_cats is not None
                    dist = TorchDistributionWithDomain(
                        distribution=torch.distributions.Categorical(
                            probs=torch.ones(n_cats, device=device) / n_cats,
                            validate_args=False,
                        ),
                        domain=domain,
                    )
                    distributions.append(dist)
                else:
                    distributions.append(UNIT_UNIFORM_DIST)

                continue

            center, conf = center_conf
            assert 0 <= conf <= 1

            # If categorical, treat it as a weighted distribution over integers
            if domain.is_categorical:
                domain_as_ints = domain.as_integer_domain()
                assert domain_as_ints.cardinality is not None

                weight_for_choice = conf
                remaining_weight = 1 - weight_for_choice

                distributed_weight = remaining_weight / (domain_as_ints.cardinality - 1)
                weights = torch.full(
                    (domain_as_ints.cardinality,),
                    distributed_weight,
                    device=device,
                    dtype=torch.float64,
                )
                center_index = domain_as_ints.cast_one(center, frm=domain)
                weights[int(center_index)] = conf

                dist = TorchDistributionWithDomain(
                    distribution=torch.distributions.Categorical(
                        probs=weights, validate_args=False
                    ),
                    domain=domain,
                )
                distributions.append(dist)
                continue

            # Otherwise, we use a continuous truncnorm
            unit_center = domain.to_unit_one(center)
            scale = torch.tensor(1 - conf, device=device, dtype=torch.float64)
            a = torch.tensor(0.0, device=device, dtype=torch.float64)
            b = torch.tensor(1.0, device=device, dtype=torch.float64)
            dist = TorchDistributionWithDomain(
                distribution=TruncatedNormal(
                    loc=unit_center,
                    scale=scale,
                    a=a,
                    b=b,
                    device=device,
                    validate_args=False,
                ),
                domain=Domain.unit_float(),
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
    [`Prior.from_parameters()`][neps.sampling.Prior.from_parameters] to use
    the `.prior` values of the parameters in a search space.
    """

    distributions: list[TorchDistributionWithDomain]
    """Distributions along with the corresponding domains they sample from."""

    _distribution_domains: list[Domain] = field(init=False)

    # OPTIM: These are used for an optimization in `log_pdf` where we do not need to
    # calculate the log_pdf of elements that have a uniform prior.
    _meaningful_ixs: list[int] = field(init=False)
    _meaningful_doms: list[Domain] = field(init=False)
    _meaningful_dists: list[Distribution] = field(init=False)

    def __post_init__(self) -> None:
        self._distribution_domains = [dist.domain for dist in self.distributions]

        rest: list[tuple[int, Domain, Distribution]] = []
        for i, dist in enumerate(self.distributions):
            if dist != UNIT_UNIFORM_DIST:
                rest.append((i, dist.domain, dist.distribution))

        if len(rest) == 0:
            self._meaningful_ixs = []
            self._meaningful_doms = []
            self._meaningful_dists = []
            return

        self._meaningful_ixs, self._meaningful_doms, self._meaningful_dists = zip(  # type: ignore
            *rest, strict=True
        )

    @property
    @override
    def ncols(self) -> int:
        return len(self.distributions)

    @override
    def log_pdf(
        self, x: torch.Tensor, *, frm: list[Domain] | Domain | ConfigEncoder
    ) -> torch.Tensor:
        if x.ndim == 0:
            raise ValueError("Expected a tensor of shape (..., ncols).")

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != len(self.distributions):
            raise ValueError(
                f"Got a tensor `x` whose last dimesion (the hyperparameter dimension)"
                f" is of length {x.shape[-1]=} but"
                f" the CenteredPrior called has {len(self.distributions)=}"
                " distributions to use for calculating the `log_pdf`. Perhaps"
                " the config or the prior have a mismatch as one includes a"
                " fidelity?"
            )

        # OPTIM: We can actually just skip elements that are distributed uniformly as
        # **assuming** they are all correctly in bounds, their log_pdf will be 0 and
        # contribute nothing.
        # It also helps numeric stability to avoid useless computations.
        if len(self._meaningful_ixs) == 0:
            return torch.zeros(x.shape[:-1], dtype=torch.float64, device=x.device)

        match frm:
            case Domain():
                pass
            case ConfigEncoder():
                frm = [frm.domains[i] for i in self._meaningful_ixs]
            case Sequence():
                frm = [frm[i] for i in self._meaningful_ixs]
            case _:
                raise TypeError(f"Unexpected type {type(frm)=}")

        # Cast all values from the value domains to the domain of the sampler.

        translated_x = Domain.translate(
            x[..., self._meaningful_ixs],
            frm=frm,
            to=self._meaningful_doms,
        )

        # Calculate the log probabilities of the sample domain tensors under their
        # respective distributions.
        # NOTE: There's no gaurantee these are actually probabilities and so we
        # treat them as unnormalized log pdfs
        itr = iter(
            zip(
                range(len(self._meaningful_dists)),
                self._meaningful_dists,
                strict=False,
            )
        )
        first_i, first_dist = next(itr)
        log_pdfs = first_dist.log_prob(translated_x[..., first_i])

        for i, dist in itr:
            log_pdfs = log_pdfs + dist.log_prob(translated_x[..., i])

        return log_pdfs

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain] | ConfigEncoder,
        seed: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        _out_shape = (
            torch.Size((n, self.ncols))
            if isinstance(n, int)
            else torch.Size((*n, self.ncols))
        )
        _n = torch.Size((n,)) if isinstance(n, int) else n

        out = torch.empty(_out_shape, device=device, dtype=dtype)
        for i, dist in enumerate(self.distributions):
            out[..., i] = dist.distribution.sample(_n)

        return Domain.translate(out, frm=self._distribution_domains, to=to, dtype=dtype)


@dataclass
class Uniform(Prior):
    """A prior that is uniform over a given domain.

    Uses a UnitUniform under the hood before converting to the value domain.
    """

    ndim: int
    """The number of columns in the tensor to sample from."""

    @property
    @override
    def ncols(self) -> int:
        return self.ndim

    @override
    def log_pdf(
        self,
        x: torch.Tensor,
        *,
        frm: Domain | list[Domain] | ConfigEncoder,
    ) -> torch.Tensor:
        # NOTE: We just assume everything is in bounds...
        shape = x.shape[:-1]  # Select everything up to last dimension (configuration)
        return torch.zeros(shape, dtype=torch.float64, device=x.device)

    @override
    def sample(
        self,
        n: int | torch.Size,
        *,
        to: Domain | list[Domain] | ConfigEncoder,
        seed: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            raise NotImplementedError("Seeding is not yet implemented.")

        _n = (
            torch.Size((n, self.ndim))
            if isinstance(n, int)
            else torch.Size((*n, self.ndim))
        )
        # Doesn't like integer dtypes
        if dtype is not None and dtype.is_floating_point:
            samples = torch.rand(_n, device=device, dtype=dtype)
        else:
            samples = torch.rand(_n, device=device)

        return Domain.translate(samples, frm=Domain.unit_float(), to=to, dtype=dtype)
