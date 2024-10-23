from pytest_cases import parametrize
from neps.sampling.samplers import Sampler, Sobol, WeightedSampler, BorderSampler
from neps.sampling.priors import Prior, Uniform

import torch

from neps.search_spaces.domain import Domain


def _make_centered_prior(ndim: int) -> Prior:
    return Prior.from_domains_and_centers(
        domains=[Domain.unit_float() for _ in range(ndim)],
        centers=[(0.5, 0.5) for _ in range(ndim)],
    )


@parametrize(
    "sampler",
    [
        Sobol(ndim=3),
        BorderSampler(ndim=3),
        Uniform(ndim=3),
        # Convenence method for making a distribution around center points
        _make_centered_prior(ndim=3),
        WeightedSampler(
            [Uniform(ndim=3), _make_centered_prior(3), Sobol(ndim=3)],
            weights=torch.tensor([0.5, 0.25, 0.25]),
        ),
    ],
)
def test_sampler_samples_into_domain(sampler: Sampler) -> None:
    assert sampler.ncols == 3

    domain_to_sample_into = Domain.integer(12, 15)
    for _ in range(10):
        x = sampler.sample(
            n=5,
            to=domain_to_sample_into,
            seed=None,
        )

        assert x.shape == (5, 3)
        assert (x >= 12).all()
        assert (x <= 15).all()

    x = sampler.sample(
        n=torch.Size((2, 1)),
        to=domain_to_sample_into,
        seed=None,
    )
    assert x.shape == (2, 1, 3)
    assert (x >= 12).all()
    assert (x <= 15).all()


@parametrize(
    "prior",
    [
        Uniform(ndim=3),
        # Convenence method for making a distribution around center points
        _make_centered_prior(ndim=3),
    ],
)
def test_priors_give_positive_pdfs(prior: Prior) -> None:
    # NOTE: The uniform prior does not check that
    assert prior.ncols == 3
    domain = Domain.floating(10, 100)

    x = prior.sample(n=5, to=domain, seed=None)
    assert x.shape == (5, 3)
    assert (x >= 10).all()
    assert (x <= 100).all()

    probs = prior.pdf(x, frm=domain)
    assert (probs >= 0).all()
    assert probs.shape == (5,)

    x = prior.sample(n=torch.Size((2, 1)), to=domain, seed=None)
    assert x.shape == (2, 1, 3)
    assert (x >= 10).all()
    assert (x <= 100).all()

    probs = prior.pdf(x, frm=domain)
    assert (probs >= 0).all()
    assert probs.shape == (2, 1)
