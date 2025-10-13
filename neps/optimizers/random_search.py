from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import ImportedConfig, SampledConfig

if TYPE_CHECKING:
    from neps.sampling import Sampler
    from neps.space import ConfigEncoder, SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


@dataclass
class RandomSearch:
    """A simple random search optimizer."""

    space: SearchSpace
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
        for config in config_dicts:
            config.update(self.space.constants)
            if self.space.fidelity is not None:
                config.update(
                    {
                        key: value.upper
                        for key, value in self.space.fidelities.items()
                        if key not in config
                    }
                )

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

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        n_trials = len(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(n_trials + i + 1)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
