from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.initial_design import next_from_initial_design
from neps.optimizers.utils.util import _get_max_trial_id
from neps.space import HPOCategorical, HPOFloat, HPOInteger
from neps.state.trial import State

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict

logger = logging.getLogger(__name__)


@dataclass
class OptunaMOTPE:
    """Optuna's multi-objective TPE (MOTPE) as a NePS optimizer.

    The Optuna study is rebuilt from the NePS trials on every call, so NePS stays the
    single source of truth: resuming a run and multiple workers need no extra state.
    The sampler seed is combined with the number of known trials so consecutive calls
    do not draw the same random startup sample.

    Integers with a `log_base` are sampled by NePS as powers of that base only, so they
    are exposed to Optuna as an integer exponent (value = log_base ** exponent) to
    search exactly the same space as the other optimizers.
    The fidelity is always set to its maximum (like BO with ignore_fidelity=True).

    If an `initial_design` is given, its configs are proposed first (indexed by the
    number of known trials) and Optuna's own random startup trials are disabled
    (unless `n_startup_trials` is given in `sampler_kwargs`).
    """

    space: SearchSpace
    """The (classic) search space to use."""

    seed: int | None
    """Seed of the TPE sampler, or None for a non-deterministic run."""

    num_objectives: int
    """Number of objectives, all minimized."""

    initial_design: list[dict[str, Any]] | None = None
    """Configs to evaluate first (completed with constants and max fidelity)."""

    sampler_kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Extra arguments for `optuna.samplers.TPESampler`."""

    def __post_init__(self) -> None:
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )

        self._distributions: dict[str, Any] = {}
        # name -> base, for integers sampled as base ** exponent
        self._exponent_bases: dict[str, float] = {}
        for name, param in self.space.searchables.items():
            match param:
                case HPOInteger(log=True, log_base=base) if base is not None:
                    self._exponent_bases[name] = base
                    self._distributions[name] = IntDistribution(
                        round(math.log(param.lower, base)),
                        round(math.log(param.upper, base)),
                    )
                case HPOInteger():
                    self._distributions[name] = IntDistribution(
                        param.lower, param.upper, log=param.log
                    )
                case HPOFloat():
                    self._distributions[name] = FloatDistribution(
                        param.lower, param.upper, log=param.log
                    )
                case HPOCategorical():
                    self._distributions[name] = CategoricalDistribution(param.choices)
                case _:
                    raise TypeError(
                        f"Unsupported parameter type {type(param)} for '{name}'"
                    )

    def _to_optuna(self, config: Mapping[str, Any]) -> dict[str, Any]:
        params = {}
        for name in self._distributions:
            value = config[name]
            if name in self._exponent_bases:
                value = round(math.log(value, self._exponent_bases[name]))
            params[name] = value
        return params

    def _from_optuna(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: self._exponent_bases[name] ** value
            if name in self._exponent_bases
            else value
            for name, value in params.items()
        }

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,  # noqa: ARG002
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        if n is not None and n != 1:
            raise ValueError("OptunaMOTPE samples one configuration at a time")
        config_id = str(_get_max_trial_id(trials) + 1)

        if design := next_from_initial_design(self.initial_design, trials, 1):
            logger.info(f"OptunaMOTPE: initial design config {design[0]}")
            return SampledConfig(id=config_id, config=design[0], previous_config_id=None)

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler_kwargs = {
            "n_startup_trials": 0 if self.initial_design else len(self._distributions),
            **self.sampler_kwargs,
        }
        seed = (
            None
            if self.seed is None
            # Mix (seed, n_trials) instead of adding them, otherwise
            # seed s at trial k == seed s+1 at trial k-1
            else int(np.random.SeedSequence([self.seed, len(trials)]).generate_state(1)[0])
        )
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=seed, **sampler_kwargs),
            directions=["minimize"] * self.num_objectives,
        )
        for trial in trials.values():
            # Pending and failed trials are skipped, as optuna's TPE ignores them
            # anyway (no constant liar). Imported trials (neps.import_trials) are
            # EXTERNAL, not SUCCESS.
            if (
                trial.metadata.state not in (State.SUCCESS, State.EXTERNAL)
                or trial.report is None
            ):
                continue
            values = trial.report.objective_to_minimize
            values = list(values) if isinstance(values, Sequence) else [values]
            if len(values) != self.num_objectives:
                raise ValueError(
                    f"Trial {trial.metadata.id} has {len(values)} objectives,"
                    f" expected {self.num_objectives}"
                )
            if not all(v is not None and math.isfinite(v) for v in values):
                continue
            study.add_trial(
                optuna.trial.create_trial(
                    params=self._to_optuna(trial.config),
                    distributions=self._distributions,
                    values=values,
                )
            )
        config = {
            **self._from_optuna(study.ask(self._distributions).params),
            **self.space.constants,
            **{name: fidelity.upper for name, fidelity in self.space.fidelities.items()},
        }
        logger.info(
            f"OptunaMOTPE: sampled config {config}"
            f" ({len(study.trials) - 1} completed trials)"
        )
        return SampledConfig(id=config_id, config=config, previous_config_id=None)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        max_trial_id = _get_max_trial_id(trials)
        return [
            ImportedConfig(id=str(max_trial_id + i + 1), config=config, result=result)
            for i, (config, result) in enumerate(external_evaluations)
        ]
