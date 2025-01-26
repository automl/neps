from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.priorband import PriorBandArgs, sample_with_priorband
from neps.optimizers.utils.brackets import PromoteAction, SampleAction
from neps.sampling.samplers import Sampler

if TYPE_CHECKING:
    from neps.optimizers.utils.brackets import Bracket
    from neps.space import SearchSpace
    from neps.space.encoding import ConfigEncoder
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
        elif isinstance(trial.report.objective_to_minimize, float):
            perf = trial.report.objective_to_minimize
        elif isinstance(trial.report.objective_to_minimize, float):
            raise NotImplementedError("Multiobjective support not implemented yet.")
        else:
            raise ValueError("Unknown type of objective_to_minimize")

        id_index[i] = _id
        rungs_index[i] = _rung
        perfs[i] = perf
        configs[i] = trial.config

    id_index = pd.MultiIndex.from_arrays([id_index, rungs_index], names=["id", "rung"])
    df = pd.DataFrame(data={"config": configs, "perf": perfs}, index=id_index)
    return df.sort_index(ascending=True)


@dataclass
class BracketOptimizer:
    """Implements an optimizer over brackets.

    This is the main class behind algorithms like `"priorband"`,
    `"successive_halving"`, `"asha"`, `"hyperband"`, etc.
    """

    space: SearchSpace
    """The pipeline space to optimize over."""

    encoder: ConfigEncoder
    """The encoder to use for the pipeline space."""

    sample_prior_first: bool | Literal["highest_fidelity"]
    """Whether or not to sample the prior first.

    If set to `"highest_fidelity"`, the prior will be sampled at the highest fidelity,
    otherwise at the lowest fidelity.
    """

    eta: int
    """The eta parameter for the algorithm."""

    rung_to_fid: Mapping[int, int | float]
    """The mapping from rung to fidelity value."""

    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]
    """A function that creates the brackets from the table of trials."""

    sampler: Sampler | PriorBandArgs
    """The sampler used to generate new trials."""

    fid_min: int | float
    """The minimum fidelity value."""

    fid_max: int | float
    """The maximum fidelity value."""

    fid_name: str
    """The name of the fidelity in the space."""

    def __call__(  # noqa: PLR0912, C901
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        space = self.space
        parameters = space.searchables

        # If we have no trials, we either go with the prior or just a sampled config
        if len(trials) == 0:
            match self.sample_prior_first:
                case "highest_fidelity":  # fid_max
                    config = {
                        name: p.prior if p.prior is not None else p.center
                        for name, p in parameters.items()
                    }
                    config.update(space.constants)
                    config[self.fid_name] = self.fid_max
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"1_{rung}", config=config)
                case True:  # fid_min
                    config = {
                        name: p.prior if p.prior is not None else p.center
                        for name, p in parameters.items()
                    }
                    config.update(space.constants)
                    config[self.fid_name] = self.fid_min
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"1_{rung}", config=config)
                case False:
                    pass

        table = trials_to_table(trials=trials)

        if len(table) == 0:  # noqa: SIM108
            # Nothing there, this sample will be the first
            nxt_id = 1
        else:
            # One plus the maximum current id in the table index
            nxt_id = table.index.get_level_values("id").max() + 1  # type: ignore

        # We don't want the first highest fidelity sample ending
        # up in a bracket
        if self.sample_prior_first == "highest_fidelity":
            table = table.iloc[1:]

        # Get and execute the next action from our brackets that are not pending or done
        assert isinstance(table, pd.DataFrame)
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
                f"{self.__class__.__name__} never got a 'sample' or 'promote' action!"
                f" This likely means the implementation of {self.create_brackets}"
                " is incorrect and should have provded enough brackets, where at"
                " least one of them should have requested another sample."
                f"\nBrackets:\n{brackets}"
            )

        match next_action:
            # The bracket would like us to promote a configuration
            case PromoteAction(config=config, id=config_id, new_rung=new_rung):
                config = {
                    **config,
                    **space.constants,
                    self.fid_name: self.rung_to_fid[new_rung],
                }
                return SampledConfig(
                    id=f"{config_id}_{new_rung}",
                    config=config,
                    previous_config_id=f"{config_id}_{new_rung - 1}",
                )

            # The bracket would like us to sample a new configuration for a rung
            case SampleAction(rung=rung):
                match self.sampler:
                    case Sampler():
                        config = self.sampler.sample_config(to=self.encoder)
                        config = {
                            **config,
                            **space.constants,
                            self.fid_name: self.rung_to_fid[rung],
                        }
                        return SampledConfig(id=f"{nxt_id}_{rung}", config=config)

                    case PriorBandArgs():
                        config = sample_with_priorband(
                            table=table,
                            parameters=space.searchables,
                            rung_to_sample_for=rung,
                            fid_bounds=(self.fid_min, self.fid_max),
                            encoder=self.encoder,
                            inc_mutation_rate=self.sampler.mutation_rate,
                            inc_mutation_std=self.sampler.mutation_std,
                            eta=self.eta,
                            seed=None,  # TODO
                        )

                        config = {
                            **config,
                            **space.constants,
                            self.fid_name: self.rung_to_fid[rung],
                        }
                        return SampledConfig(id=f"{nxt_id}_{rung}", config=config)
                    case _:
                        raise RuntimeError(f"Unknown sampler: {self.sampler}")
            case _:
                raise RuntimeError(f"Unknown bracket action: {next_action}")
