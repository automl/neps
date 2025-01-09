from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neps.optimizers.optimizer import SampledConfig

if TYPE_CHECKING:
    from neps.sampling.samplers import Sampler
    from neps.search_spaces.encoding import ConfigEncoder
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


@dataclass
class RandomSearch:
    """A simple random search optimizer."""

    pipeline_space: SearchSpace
    ignore_fidelity: bool
    seed: int | None
    encoder: ConfigEncoder
    sampler: Sampler

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        n_trials = len(trials)
        _n = 1 if n is None else n
        configs = self.sampler.sample(_n, to=self.encoder.domains)
        config_dicts = self.encoder.decode(configs)
        if n == 1:
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
