from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neps.optimizers.optimizer import SampledConfig

if TYPE_CHECKING:
    from neps.sampling import Sampler
    from neps.space import ConfigEncoder, SearchSpace
    from neps.state import BudgetInfo, Trial


@dataclass
class RandomSearch:
    """A simple random search optimizer."""

    space: SearchSpace
    encoder: ConfigEncoder
    numerical_sampler: Sampler

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        n_trials = len(trials)
        _n = 1 if n is None else n
        configs_tensor: Tensor = self.numerical_sampler.sample(_n, to=self.encoder)

        config_dicts = self.encoder.decode(configs_tensor)
        for config in config_dicts:
            config.update(self.space.constants)

        # TODO: We should probably have a grammar sampler class, not do it manually here
        # This works for now but should be updated.
        if self.space.grammar is not None:
            rng = np.random.default_rng()  # TODO: We should be able to seed this.
            grammar_key, grammar = self.space.grammar
            for config in config_dicts:
                sample = grammar.sample(rng=rng)
                config.update({grammar_key: sample.to_string()})

        if n is None:
            config = config_dicts[0]
            config_id = str(n_trials + 1)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(n_trials + i + 1),
                previous_config_id=None,
            )
            for i, config in enumerate(config_dicts)
        ]
