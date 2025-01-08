from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.priorband import PriorBandArgs, sample_with_priorband
from neps.sampling.samplers import Sampler

if TYPE_CHECKING:
    from neps.optimizers.utils.brackets import Bracket
    from neps.search_spaces import SearchSpace
    from neps.search_spaces.encoding import ConfigEncoder
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


logger = logging.getLogger(__name__)


def trials_to_table(trials: Mapping[str, Trial]) -> pd.DataFrame:
    id_index = np.empty(len(trials), dtype=int)
    rungs_index = np.empty(len(trials), dtype=int)
    perfs = np.empty(len(trials), dtype=np.float64)
    configs = np.empty(len(trials), dtype=object)

    for i, (trial_id, trial) in enumerate(trials.items()):
        config_id_str, rung_str = trial_id.split("_")
        _id, _rung = int(config_id_str), int(rung_str)

        if trial.report is None:
            perf = np.nan  # Pending
        elif trial.report.objective_to_minimize is None:
            perf = np.inf  # Error? Either way, we wont promote it
        else:
            perf = trial.report.objective_to_minimize

        id_index[i] = _id
        rungs_index[i] = _rung
        perfs[i] = perf
        configs[i] = trial.config

    id_index = pd.MultiIndex.from_arrays([id_index, rungs_index], names=["id", "rung"])
    df = pd.DataFrame(data={"config": configs, "perf": perfs}, index=id_index)
    return df.sort_index(ascending=True)


@dataclass
class BracketOptimizer:
    """Implements an optimizer over brackets."""

    pipeline_space: SearchSpace
    encoder: ConfigEncoder
    sample_prior_first: bool | Literal["highest_fidelity"]
    eta: int
    rung_to_fid: Mapping[int, int | float]
    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]
    sampler: Sampler | PriorBandArgs

    fid_min: int | float
    fid_max: int | float
    fid_name: str

    def __call__(  # noqa: PLR0912, C901
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        space = self.pipeline_space

        # If we have no trials, we either go with the prior or just a sampled config
        if len(trials) == 0:
            match self.sample_prior_first:
                case "highest_fidelity":
                    config = {**space.prior_config, self.fid_name: self.fid_max}
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"0_{rung}", config=config)
                case True:
                    config = {**space.prior_config, self.fid_name: self.fid_min}
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"0_{rung}", config=config)
                case False:
                    pass

        # We have to special case this as we don't want it ending up in a bracket
        if self.sample_prior_first == "highest_fidelity":
            table = trials_to_table(trials=trials)[1:]
            assert isinstance(table, pd.DataFrame)
        else:
            table = trials_to_table(trials=trials)

        if len(table) == 0:
            nxt_id = 0
        else:
            nxt_id = int(table.index.get_level_values("id").max()) + 1  # type: ignore

        # Get and execute the next action from our brackets that are not pending or done
        brackets = self.create_brackets(table)
        if not isinstance(brackets, Sequence):
            brackets = [brackets]

        next_action = next(
            (
                action
                for bracket in brackets
                if (action := bracket.next()) not in ("done", "pending")
            ),
            None,
        )

        if next_action is None:
            raise RuntimeError(
                f"{self.__class__.__name__} never got a 'sample' or 'pending' action!"
            )

        match next_action:
            case ("promote", config, config_id, new_rung):
                config = {**config, self.fid_name: self.rung_to_fid[new_rung]}
                return SampledConfig(
                    id=f"{config_id}_{new_rung}",
                    config=config,
                    previous_config_id=f"{config_id}_{new_rung - 1}",
                )
            case ("new", rung):
                match self.sampler:
                    case Sampler():
                        config = self.sampler.sample_config(to=self.encoder)
                        config = {**config, self.fid_name: self.rung_to_fid[rung]}
                        return SampledConfig(id=f"{nxt_id}_{rung}", config=config)
                    case PriorBandArgs():
                        config = sample_with_priorband(
                            table=table,
                            space=space,
                            rung_to_sample_for=rung,
                            fid_bounds=(self.fid_min, self.fid_max),
                            encoder=self.encoder,
                            inc_mutation_rate=self.sampler.mutation_rate,
                            inc_mutation_std=self.sampler.mutation_std,
                            eta=self.eta,
                            seed=None,  # TODO
                        )
                        config = {**config, self.fid_name: self.rung_to_fid[rung]}
                        return SampledConfig(id=f"{nxt_id}_{rung}", config=config)
                    case _:
                        raise RuntimeError(f"Unknown sampler: {self.sampler}")
            case _:
                raise RuntimeError(f"Unknown bracket action: {next_action}")
