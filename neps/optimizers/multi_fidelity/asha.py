from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import numpy as np

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.multi_fidelity.brackets import AsyncBracket
from neps.optimizers.multi_fidelity.utils import trials_to_table
from neps.sampling.priors import Prior, Uniform
from neps.sampling.samplers import WeightedSampler
from neps.search_spaces import Integer, SearchSpace
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial

logger = logging.getLogger(__name__)


class ASHA(BaseOptimizer):
    """Implements a ASHA procedure."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        eta: int = 3,
        early_stopping_rate: int = 0,
        sampler: (
            Literal["uniform", "prior"] | Mapping[Literal["uniform", "prior"], float]
        ) = "uniform",
        sample_default_first: bool | Literal["highest_fidelity"] = False,
        # TODO: Remove
        budget: int | float | None = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
    ):
        """Initialise an SH bracket.

        Args:
            pipeline_space: Space in which to search
            eta: The reduction factor used by SH
            early_stopping_rate: Determines the number of rungs in an SH bracket
                Choosing 0 creates maximal rungs given the fidelity bounds
            sampler: The type of sampling procedure to use:

                * If "uniform", samples uniformly from the space when it needs to sample
                * If "prior", samples from the prior distribution built from the default
                  and default_confidence values in the pipeline space.
                * If a dict, it should include both "uniform" and "prior" keys along
                    with their weights.

            sample_default_first: Whether to sample the default configuration first.
        """
        if len(pipeline_space.fidelities) != 1:
            raise ValueError(
                "1 Fidelity should be defined in the pipeline space for"
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

        match sampler:
            case "uniform":
                _sampler = Uniform.from_space(pipeline_space)
            case "prior":
                _sampler = Prior.from_space(pipeline_space)
            case {"uniform": u, "prior": p}:
                _sampler = WeightedSampler(
                    samplers=[
                        Uniform.from_space(pipeline_space),
                        Prior.from_space(pipeline_space),
                    ],
                    weights=[u, p],
                )
            case _:
                raise ValueError(
                    f"Invalid sampler: {sampler}. Please provide either 'uniform' or"
                    " 'prior' or a dict with both keys and their weights."
                )

        # Samler and corresponding encoder/decoder for it
        self.sampler = _sampler
        self.encoder = ConfigEncoder.from_space(pipeline_space, include_fidelity=False)

        self.eta = eta
        self.sample_default_first: bool | Literal["highest_fidelity"] = (
            sample_default_first
        )

        _fid_name = pipeline_space.fidelity_name
        assert _fid_name is not None
        assert pipeline_space.fidelity is not None

        bmin: int | float = pipeline_space.fidelity.lower
        bmax: int | float = pipeline_space.fidelity.upper
        budget_type = int if isinstance(pipeline_space.fidelity, Integer) else float
        self._fid_name: str = _fid_name
        self._fid_max: int | float = bmax

        # Ensure no rung ID is negative, s_max in https://arxiv.org/pdf/1603.06560.pdf
        esr = early_stopping_rate
        stop_rate_limit = int(np.floor(np.log(bmax / bmin) / np.log(eta)))
        assert early_stopping_rate <= stop_rate_limit

        # maps rungs to a fidelity value for an SH bracket with `early_stopping_rate`
        nrungs = int(np.floor(np.log(bmax / (bmin * (eta**esr))) / np.log(eta)) + 1)

        self.rung_to_fidelity = {
            esr + j: budget_type(bmax / (self.eta**i))
            for i, j in enumerate(reversed(range(nrungs)))
        }
        self.min_rung = min(self.rung_to_fidelity)
        self.max_rung = max(self.rung_to_fidelity)

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        _min_fid = {self._fid_name: self.rung_to_fidelity[self.min_rung]}
        space = self.pipeline_space

        # If we have no trials, we either go with the default or just a sampled config
        if len(trials) == 0:
            match self.sample_default_first:
                case "highest_fidelity":
                    config = {**space.default_config, self._fid_name: self._fid_max}
                    rung = self.max_rung
                case True:
                    config = {**space.default_config, **_min_fid}
                    rung = self.min_rung
                case False:
                    config = self.sampler.sample_config(to=self.encoder, extra=_min_fid)
                    rung = self.min_rung
                case _:
                    raise RuntimeError("This is a bug!")

            return SampledConfig(id=f"0_{rung}", config=config)

        table = trials_to_table(trials=trials)

        # We have to special case this as we don't want it ending up in a bracket
        if self.sample_default_first == "highest_fidelity":
            table_for_brackets = table.iloc[1:]
        else:
            table_for_brackets = table

        bracket = AsyncBracket.make_asha_bracket(
            table=table_for_brackets, rungs=list(self.rung_to_fidelity), eta=self.eta
        )

        match bracket.next():
            case ("new", rung):
                config = self.sampler.sample_config(to=self.encoder, extra=_min_fid)
                _id = int(table.index.get_level_values("id").max()) + 1  # type: ignore
                return SampledConfig(id=f"{_id}_{rung}", config=config)

            case ("promote", config, _id, new_rung):
                config = {**config, **_min_fid}
                return SampledConfig(
                    id=f"{_id}_{new_rung}",
                    config=config,
                    previous_config_id=f"{_id}_{new_rung - 1}",
                )

            case _:
                raise RuntimeError("This is a bug!")
