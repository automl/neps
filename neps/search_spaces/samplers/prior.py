from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping
from typing_extensions import Self, override

from neps.search_spaces.config import Config
from neps.search_spaces.distributions.uniform_int import UniformIntDistribution
from neps.search_spaces.distributions.weighted_ints import WeightedIntsDistribution
from neps.search_spaces.samplers.sampler import Sampler

if TYPE_CHECKING:
    import numpy as np

    from neps.search_spaces.distributions.distribution import Distribution
    from neps.search_spaces.search_space import SearchSpace


@dataclass
class PriorSampler(Sampler):
    search_space: SearchSpace

    _numerical_distributions: Mapping[str, Distribution]
    _categorical_distributions: Mapping[str, Distribution]

    @override
    def sample_configs(
        self,
        n: int,
        *,
        fidelity: Mapping[str, float] | None,
        seed: np.random.Generator,
        with_constants: bool = True,
    ) -> list[Config]:
        numerical_samples = {}
        for k, dist in self._numerical_distributions.items():
            param = self.search_space.numericals[k]
            numerical_samples[k] = dist.sample(n, to=param.domain, seed=seed)

        categorical_samples = {}
        for k, dist in self._categorical_distributions.items():
            cat = self.search_space.categoricals[k]
            domain = cat.domain
            samples = dist.sample(n, to=domain, seed=seed)
            choices = cat.lookup(samples)
            categorical_samples[k] = choices

        graph_samples = {}
        for k, v in self.search_space.graphs.items():
            graph_samples[k] = [v.sample() for _ in range(n)]

        _constants = self.search_space.constants if with_constants else {}

        return [
            Config(
                values={
                    **{k: samples[i] for k, samples in numerical_samples.items()},
                    **{k: samples[i] for k, samples in categorical_samples.items()},
                    **{k: samples[i] for k, samples in graph_samples.items()},
                    **_constants,
                },
                fidelity=fidelity,
            )
            for i in range(n)
        ]

    @classmethod
    def new(
        cls,
        space: SearchSpace,
        prior: Mapping[str, tuple[Any, float]],
        *,
        replace_missing_with_uniform: bool = True,
    ) -> Self:
        missing = set(space.hyperparameters) - set(prior.keys())
        if not replace_missing_with_uniform and any(missing):
            raise ValueError(
                "If `replace_missing_with_uniform` is False, the prior must be defined"
                f" for all parameters. Missing prior for: {missing}"
            )

        numerical_distributions = {
            hp_name: (
                hp.domain.truncnorm_distribution(center=p[0], confidence=p[1])
                if (p := prior.get(hp_name))
                else hp.domain.uniform_distribution()
            )
            for hp_name, hp in space.numericals.items()
        }
        # NOTE: It would be nice to somehow check if the prior given for
        # a categorical was an index or a value in the categorical.
        # Since it's much more efficient to hold on to the index, we will
        # assume that for now.
        categorical_distribution = {
            hp_name: (
                WeightedIntsDistribution.with_favoured(
                    n=cat.size,
                    favoured=cat.index(p[0]),
                    confidence=p[1],
                )
                if (p := prior.get(hp_name))
                else UniformIntDistribution.indices(cat.size)
            )
            for hp_name, cat in space.categoricals.items()
        }
        return cls(
            space,
            _numerical_distributions=numerical_distributions,
            _categorical_distributions=categorical_distribution,
        )
