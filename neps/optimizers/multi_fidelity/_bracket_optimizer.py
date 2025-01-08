from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.multi_fidelity.priorband import PriorBandArgs, sample_with_priorband
from neps.sampling.priors import Prior
from neps.sampling.samplers import Sampler
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from neps.optimizers.multi_fidelity.brackets import Bracket
    from neps.search_spaces import SearchSpace
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
        elif trial.report.loss is None:
            perf = np.inf  # Error? Either way, we wont promote it
        else:
            perf = trial.report.loss

        id_index[i] = _id
        rungs_index[i] = _rung
        perfs[i] = perf
        configs[i] = trial.config

    id_index = pd.MultiIndex.from_arrays([id_index, rungs_index], names=["id", "rung"])
    df = pd.DataFrame(data={"config": configs, "perf": perfs}, index=id_index)
    return df.sort_index(ascending=True)


class BracketOptimizer(BaseOptimizer):
    """Implements an optimizer over brackets."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        eta: int,
        sampler: Literal["uniform", "prior", "priorband"] | PriorBandArgs | Sampler,
        sample_default_first: bool | Literal["highest_fidelity"],
        create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket],
        rung_to_fidelity: Mapping[int, int | float],
        # TODO: Remove
        budget: int | float | None = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
    ):
        """Initialise a bracket optimizer.

        Args:
            pipeline_space: Space in which to search
            eta: The reduction factor used for building brackets
            sampler: The type of sampling procedure to use:

                * If "uniform", samples uniformly from the space when it needs to sample
                * If "prior", samples from the prior distribution built from the default
                  and default_confidence values in the pipeline space.
                * If [PriorBandArgs][neps.optimizers.multi_fidelity.priorband],
                    samples with weights according to the PriorBand algorithm.
                    See: https://arxiv.org/abs/2306.12370

            sample_default_first: Whether to sample the default configuration first.
        """
        if len(pipeline_space.fidelities) != 1:
            raise ValueError(
                "Fidelity should be defined in the pipeline space for"
                f" {self.__class__.__name__} to work. Got: {pipeline_space.fidelities}"
            )

        if sample_default_first not in (True, False, "highest_fidelity"):
            raise ValueError(
                "sample_default_first should be either True, False or 'highest_fidelity'"
            )

        super().__init__(
            pipeline_space=pipeline_space,
            budget=None,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
        )

        space = self.pipeline_space
        self.encoder = ConfigEncoder.from_space(space, include_fidelity=False)
        self.sample_default_first = sample_default_first
        self.eta = eta
        self.rung_to_fid = rung_to_fidelity
        self.create_brackets = create_brackets

        assert pipeline_space.fidelity is not None
        assert pipeline_space.fidelity_name is not None
        self.fid_min: int | float = pipeline_space.fidelity.lower
        self.fid_max: int | float = pipeline_space.fidelity.upper
        self.fid_name: str = pipeline_space.fidelity_name

        match sampler:
            case "uniform":
                _sampler = Sampler.uniform(ndim=self.encoder.ndim)
            case "prior":
                _sampler = Prior.from_config(pipeline_space.default_config, space=space)
            case "priorband":
                _sampler = PriorBandArgs(mutation_rate=0.5, mutation_std=0.25)
            case PriorBandArgs() | Sampler():
                _sampler = sampler
            case _:
                raise ValueError(f"Unknown sampler: {sampler}")

        self.sampler: Sampler | PriorBandArgs = _sampler

    @override
    def ask(  # noqa: PLR0912, C901
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        space = self.pipeline_space

        # If we have no trials, we either go with the default or just a sampled config
        if len(trials) == 0:
            match self.sample_default_first:
                case "highest_fidelity":
                    config = {**space.default_config, self.fid_name: self.fid_max}
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"0_{rung}", config=config)
                case True:
                    config = {**space.default_config, self.fid_name: self.fid_min}
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"0_{rung}", config=config)
                case False:
                    pass

        # We have to special case this as we don't want it ending up in a bracket
        if self.sample_default_first == "highest_fidelity":
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
