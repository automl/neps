from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping
from typing_extensions import Self, override

import numpy as np

from neps.search_spaces.samplers.sampler import Sampler
from neps.utils.types import Arr, Number, f64

if TYPE_CHECKING:
    from neps.search_spaces.config import Config


@dataclass
class WeightedSampler(Sampler):
    weights: dict[str, float]
    samplers: dict[str, Sampler]

    _probabilities: Arr[f64] = field(init=False, repr=False, compare=False)
    _samplers: Arr[np.str_] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        probs = np.array(list(self.weights.values()), dtype=f64)
        probs /= probs.sum()
        self._probabilities = probs
        self._samplers = np.asarray(sorted(self.samplers.keys()), dtype=np.str_)

    @override
    def sample_configs(
        self,
        n: int,
        *,
        fidelity: Mapping[str, Number] | None,
        seed: np.random.Generator,
    ) -> list[Config]:
        choices = seed.choice(self._samplers, size=n, p=self._probabilities)
        keys, counts = np.unique(choices, return_counts=True)

        configs: list[Config] = []
        for key, count in zip(keys, counts):
            sampler = self.samplers[key]
            config_samples = sampler.sample_configs(count, fidelity=fidelity, seed=seed)
            configs.extend(config_samples)

        return configs

    @classmethod
    def equally_weighted(cls, samples: dict[str, Sampler]) -> Self:
        return cls(weights={k: 1.0 for k in samples}, samplers=samples)
