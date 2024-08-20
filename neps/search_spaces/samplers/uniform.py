from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
from typing_extensions import Self, override

from neps.search_spaces.config import Config
from neps.search_spaces.distributions.uniform_int import UniformIntDistribution
from neps.search_spaces.samplers.sampler import Sampler

if TYPE_CHECKING:
    import numpy as np

    from neps.search_spaces.distributions.distribution import Distribution
    from neps.search_spaces.search_space import SearchSpace


@dataclass
class UniformSampler(Sampler):
    search_space: SearchSpace

    _numerical_distributions: Mapping[str, Distribution]
    _categorical_distributions: Mapping[str, Distribution]

    @override
    def sample_configs(
        self,
        n: int,
        *,
        fidelity: Mapping[str, float] | None = None,
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
                {
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
    def new(cls, space: SearchSpace) -> Self:
        numerical_distributions = {
            k: p.domain.uniform_distribution() for k, p in space.numericals.items()
        }
        categorical_distribution = {
            k: UniformIntDistribution.indices(p.size)
            for k, p in space.categoricals.items()
        }
        return cls(
            space,
            _numerical_distributions=numerical_distributions,
            _categorical_distributions=categorical_distribution,
        )
