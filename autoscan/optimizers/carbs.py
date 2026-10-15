"""CARBS is an external library, install it with `pip install auto-scan[carbs]`.

CARBS keeps its observations in memory and checkpoints them to its own directory. Here
it is stateless instead: a new CARBS instance is built on every ask and replays the
trials AutoScAn hands in, so the trials in `root_directory` stay the only record of the
run and several workers, or a continued run, all see the same history.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    Param,
)
from carbs.utils import SUGGESTION_ID_DICT_KEY, SuggestionInBasic
from loguru import logger as carbs_logger

from autoscan.optimizers.optimizer import ImportedConfig, SampledConfig, ScalingOptimizer
from autoscan.optimizers.utils.util import _get_next_trial_id
from autoscan.space.parameters import HPOFloat, HPOInteger, Parameter
from autoscan.state.seed_snapshot import SeedSnapshot

if TYPE_CHECKING:
    from autoscan.space import SearchSpace
    from autoscan.state import BudgetInfo, Trial
    from autoscan.state.pipeline_eval import UserResultDict

logger = logging.getLogger(__name__)


def _center(hp: HPOFloat | HPOInteger, lower: float, upper: float) -> float:
    """The middle of `[lower, upper]` on the scale the parameter is searched on."""
    if not hp.log:
        center = (lower + upper) / 2
    elif hp.log_base is None:
        center = math.sqrt(lower * upper)
    else:
        # Stay on a power of the base, rounding down so that shrinking reaches `lower`
        exponent_lower = round(math.log(lower, hp.log_base))
        exponent_upper = round(math.log(upper, hp.log_base))
        center = hp.log_base ** ((exponent_lower + exponent_upper) // 2)
    return int(center) if isinstance(hp, HPOInteger) else center


def convert_space_to_carbs_space(
    space: Mapping[str, Parameter],
) -> Iterator[Param]:
    """Convert numerical parameters to CARBS parameters with a search center."""
    for name, hp in space.items():
        if not isinstance(hp, HPOInteger | HPOFloat):
            raise NotImplementedError(
                f"CARBS only supports Integer and Float parameters, but '{name}' is"
                f" a {type(hp).__name__}."
            )

        is_integer = isinstance(hp, HPOInteger)
        lower, upper = hp.domain.lower, hp.domain.upper
        search_center = hp.prior if hp.prior is not None else _center(hp, lower, upper)
        if hp.log:
            base = hp.log_base if hp.log_base is not None else 10
            yield Param(
                name=name,
                space=LogSpace(
                    scale=math.log(math.e, base),
                    min=lower,
                    max=upper,
                    is_integer=is_integer,
                    base=base,
                ),
                search_center=search_center,
            )
        elif lower >= 0 and upper <= 1:
            yield Param(
                name=name,
                space=LogitSpace(scale=1.0, min=lower, max=upper, is_integer=is_integer),
                search_center=search_center,
            )
        else:
            yield Param(
                name=name,
                space=LinearSpace(
                    scale=3 * (upper - lower),
                    min=lower,
                    max=upper,
                    is_integer=is_integer,
                ),
                search_center=search_center,
            )


@dataclass
class CARBSOptimizer(ScalingOptimizer):
    """Suggests configurations with CARBS, rebuilt from the trials on every ask.

    The cost of an evaluation is the `cost` reported for it. If none is reported, the
    `cost_estimator` given to `autoscan.run()` is used.
    """

    space: SearchSpace
    """The search space, only Integer and Float parameters are supported."""

    initial_search_radius: float = 0.3
    """The standard deviation, in CARBS' search space, to sample around a center."""

    num_random_samples: int = 4
    """Sample around the search center until this many evaluations succeeded."""

    resample_frequency: int = 0
    """Re-evaluate a pareto-optimal configuration every this many evaluations."""

    search_center_cost_fraction: float | None = None
    """Move the search center of `search_center_shrink_param` towards its lower bound
    until the center costs at most this fraction of `max_cost_per_config`. Each step
    takes the middle of the lower bound and the center, on the scale the parameter is
    searched on."""

    search_center_shrink_param: str | None = None
    """The parameter to shrink for `search_center_cost_fraction`."""

    def __post_init__(self) -> None:
        if self.space.fidelities:
            raise ValueError(
                "CARBS does not support fidelities, but the pipeline space has"
                f" {list(self.space.fidelities)}."
            )
        # Fail on unsupported parameters when the optimizer is created, not on first ask
        list(convert_space_to_carbs_space(self.space.searchables))

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        carbs_opt = self._replay(trials)
        next_id = _get_next_trial_id(trials)

        sampled_configs: list[SampledConfig] = []
        for i in range(1 if n is None else n):
            # CARBS remembers what it suggested, so a batch does not repeat itself
            suggestion = dict(carbs_opt.suggest().suggestion)
            suggestion.pop(SUGGESTION_ID_DICT_KEY, None)
            for name, hp in self.space.searchables.items():
                assert isinstance(hp, HPOInteger | HPOFloat)  # Checked on creation
                value = min(max(suggestion[name], hp.lower), hp.upper)
                suggestion[name] = int(value) if isinstance(hp, HPOInteger) else value

            config = {**self.space.constants, **suggestion}
            sampled_configs.append(SampledConfig(id=str(next_id + i), config=config))

        return sampled_configs[0] if n is None else sampled_configs

    def _param_spaces(self) -> list[Param]:
        param_spaces = list(convert_space_to_carbs_space(self.space.searchables))
        fraction, name = self.search_center_cost_fraction, self.search_center_shrink_param
        if fraction is None:
            return param_spaces

        shrink = next((p for p in param_spaces if p.name == name), None)
        if self.cost_estimator is None or self.max_cost_per_config is None:
            raise ValueError(
                "`search_center_cost_fraction` needs `cost_estimator` and"
                " `max_cost_per_config` to be given to `autoscan.run()`."
            )
        if shrink is None:
            raise ValueError(
                "`search_center_shrink_param` must be one of"
                f" {list(self.space.searchables)}, got '{name}'."
            )

        # Move the parameter's search center towards its lower bound, on the scale it is
        # searched on, until the center fits the cost target
        assert name is not None  # `shrink` was found by it
        hp = self.space.searchables[name]
        assert isinstance(hp, HPOInteger | HPOFloat)  # Checked on creation
        target = fraction * self.max_cost_per_config
        probe = {
            **self.space.constants,
            **{p.name: p.search_center for p in param_spaces},
        }
        before = probe[name]
        while self.cost_estimator(**probe) > target:
            shrunk = _center(hp, hp.lower, probe[name])
            if not hp.lower <= shrunk < probe[name]:
                break
            probe[name] = shrunk
        if probe[name] == before:
            return param_spaces

        logger.debug(
            "CARBS search center: %s %.4g -> %.4g to cost at most %.3e (%s x"
            " max_cost_per_config)",
            name,
            before,
            probe[name],
            target,
            fraction,
        )
        return [
            Param(name=p.name, space=p.space, search_center=probe[name])
            if p.name == name
            else p
            for p in param_spaces
        ]

    def _replay(self, trials: Mapping[str, Trial]) -> CARBS:
        params = CARBSParams(
            better_direction_sign=-1,
            num_random_samples=self.num_random_samples,
            initial_search_radius=self.initial_search_radius,
            resample_frequency=self.resample_frequency,
            max_suggestion_cost=self.max_cost_per_config,
            is_wandb_logging_enabled=False,
            is_saved_on_every_observation=False,
        )
        # CARBS seeds the global RNGs when it is created. As that happens on every ask,
        # put back the state AutoScAn manages, the random phase would repeat otherwise.
        rng_state = SeedSnapshot.new_capture()
        carbs_logger.disable("carbs")  # Logs its whole config on creation
        try:
            carbs_opt = CARBS(params, self._param_spaces())
        finally:
            carbs_logger.enable("carbs")
            rng_state.set_as_global_seed_state()

        for trial in trials.values():
            carbs_input = {name: trial.config[name] for name in self.space.searchables}
            if trial.report is None:
                # Still pending, CARBS accounts for it to not suggest the same again
                suggestion = SuggestionInBasic(
                    carbs_opt._param_space_real_to_basic_space_real(carbs_input)
                )
                carbs_opt.outstanding_suggestions[trial.id] = suggestion
            else:
                carbs_opt.observe(self._trial_to_observation(trial, carbs_input))

        return carbs_opt

    def _trial_to_observation(
        self, trial: Trial, carbs_input: dict[str, Any]
    ) -> ObservationInParam:
        report = trial.report
        assert report is not None
        objective = report.objective_to_minimize
        if isinstance(objective, Sequence):
            raise ValueError(
                "CARBS only supports a single objective, but trial"
                f" '{trial.id}' reported {objective}."
            )

        over_budget = (
            self.cost_estimator is not None
            and self.max_cost_per_config is not None
            and self.cost_estimator(**trial.config) > self.max_cost_per_config
        )
        if (
            report.reported_as != "success"
            or objective is None
            or not math.isfinite(objective)
            or over_budget
        ):
            return ObservationInParam(
                input=carbs_input, output=float("inf"), cost=0.0, is_failure=True
            )

        cost = report.cost
        if cost is None and self.cost_estimator is not None:
            cost = self.cost_estimator(**trial.config)
        if cost is None:
            raise ValueError(
                f"CARBS needs the cost of every evaluation, but trial '{trial.id}'"
                " has none. Return a `cost` from `evaluate_pipeline` or pass a"
                " `cost_estimator` to `autoscan.run()`."
            )
        return ObservationInParam(input=carbs_input, output=objective, cost=cost)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        # Imported trials are replayed like any other one on the next ask
        next_id = _get_next_trial_id(trials)
        return [
            ImportedConfig(id=str(next_id + i), config=config, result=result)
            for i, (config, result) in enumerate(external_evaluations)
        ]

    def get_trial_artifacts(self, trials: Mapping[str, Trial] | None = None) -> None:
        return None
