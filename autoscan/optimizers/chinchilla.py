"""Chinchilla guided scaling, see https://arxiv.org/abs/2203.15556.

The configurations to evaluate are laid out once, on the first ask, from the
`cost_estimator`, `max_cost_per_config` and cost budget given to `autoscan.run()`. They
are then evaluated in order, like a grid search.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch

from autoscan.optimizers.grid_search import GridSearch
from autoscan.optimizers.optimizer import (
    Artifact,
    ArtifactType,
    ImportedConfig,
    SampledConfig,
    ScalingOptimizer,
)
from autoscan.sampling import Sampler
from autoscan.space import Domain, HPOCategorical
from autoscan.utils.cost_estimator import CostEstimator
from autoscan.utils.pareto import pareto_efficient
from autoscan.utils.power_law import LogLogPowerLawFit, fit_power_law

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from autoscan.space import SearchSpace
    from autoscan.state import BudgetInfo, Trial
    from autoscan.state.pipeline_eval import UserResultDict

logger = logging.getLogger(__name__)

ModelConfigs: TypeAlias = defaultdict[float, list[tuple[tuple[str, Any], ...]]]
"""Model configurations, without the data parameter, by their parameter count."""


def _power_law_curve(fit: LogLogPowerLawFit, xs: Sequence[float]) -> np.ndarray:
    return 10.0 ** (fit.slope * np.log10(xs) + fit.intercept)


@dataclass(frozen=True, slots=True)
class _ScalingPoint:
    """One evaluation, or one point of its learning curve."""

    n_params: float
    n_data: float
    loss: float
    flops: float


def _learning_curve_points(learning_curve: Any, n_params: float) -> list[_ScalingPoint]:
    """The points of a learning curve reported as rows of (n_data, flops, loss)."""
    return [
        _ScalingPoint(
            n_params=n_params,
            n_data=float(row[0]),
            flops=float(row[1]),
            loss=float(row[-1]),
        )
        for row in learning_curve or []
        if isinstance(row, Sequence) and len(row) >= 3
    ]


def _enforce_monotonic_params(points: list[_ScalingPoint]) -> list[_ScalingPoint]:
    """Drop the points whose n_params rises as flops falls.

    Walks down from the largest-flops point and returns the kept points sorted by
    ascending flops.
    """
    if len(points) <= 1:
        return points

    descending = sorted(points, key=lambda point: point.flops, reverse=True)
    monotonic = [descending[0]]
    for point in descending[1:]:
        if point.n_params <= monotonic[-1].n_params:
            monotonic.append(point)
    return monotonic[::-1]


def _pareto_front(points: list[_ScalingPoint]) -> list[_ScalingPoint]:
    """Pareto front of loss vs flops with n_params monotone in flops, ascending flops."""
    on_front = pareto_efficient(
        np.array([(point.flops, point.loss) for point in points]).reshape(-1, 2)
    )
    return _enforce_monotonic_params(
        [point for point, keep in zip(points, on_front, strict=True) if keep]
    )


def _create_scaling_plot(
    xs: list, ys: list, xlabel: str, ylabel: str, title: str
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(xs, ys, marker="o", alpha=0.9)

    fit = fit_power_law(xs, ys)
    if fit is not None:
        ax.plot(
            xs,
            _power_law_curve(fit, xs),
            "r--",
            linewidth=2,
            alpha=0.8,
            label=f"Fit (slope={fit.slope:.3f})",
        )
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    return fig


@dataclass
class ChinchillaGuidedScaling(ScalingOptimizer):
    """Evaluates a list of configurations laid out along the scaling parameters.

    The parameters marked with `is_compute_scaling=True` make up the model, the one
    named `data_parameter_name` the amount of data. Every other parameter is drawn at
    random once and held fixed.
    """

    space: SearchSpace
    """The search space, fidelities are not supported."""

    data_parameter_name: str
    """The parameter that sets the amount of training data."""

    params_estimator: Callable[..., float]
    """Called as `params_estimator(**config)` for the parameter count of the model. A
    function without `**kwargs` is only passed the parameters it names."""

    strategy: Literal["approach1", "approach2"] = "approach2"
    """`"approach1"` trains every model on `data_multipliers` times its parameter count,
    `"approach2"` places configurations on ISO-FLOP levels."""

    data_multipliers: Sequence[float] | None = None
    """The data to parameter count ratios of `"approach1"`."""

    iso_flop_steps: int = 10
    """The number of ISO-FLOP levels of `"approach2"`."""

    candidates_per_iso_flop: int = 10
    """How many configurations to place on an ISO-FLOP level."""

    base_optimizer: GridSearch | None = field(default=None, init=False, repr=False)
    """Evaluates the laid out configurations, `None` until the first ask."""

    def __post_init__(self) -> None:
        # Like the cost estimator, only passed the parameters it names
        self._n_params = CostEstimator.from_callable(self.params_estimator)
        if self.space.fidelities:
            raise ValueError(
                "Chinchilla does not support fidelities, but the pipeline space has"
                f" {list(self.space.fidelities)}."
            )
        if self.strategy not in ("approach1", "approach2"):
            raise ValueError(
                f"`strategy` must be 'approach1' or 'approach2', got '{self.strategy}'."
            )
        if self.strategy == "approach1" and not self.data_multipliers:
            raise ValueError("`strategy='approach1'` needs `data_multipliers`.")
        if self.data_parameter_name not in self.space.searchables:
            raise ValueError(
                "`data_parameter_name` must be one of"
                f" {list(self.space.searchables)}, got '{self.data_parameter_name}'."
            )
        if not self._model_scaling_params():
            raise ValueError(
                "Chinchilla needs at least one parameter with `is_compute_scaling=True`"
                f" other than '{self.data_parameter_name}'."
            )

    def _model_scaling_params(self) -> dict[str, Any]:
        return {
            name: param
            for name, param in self.space.searchables.items()
            if getattr(param, "is_compute_scaling", False)
            and name != self.data_parameter_name
        }

    def _flops(self, config: Mapping[str, Any]) -> float:
        assert self.cost_estimator is not None
        return self.cost_estimator(**config)

    def _ensure_grid(self, budget_info: BudgetInfo | None) -> GridSearch:
        if self.base_optimizer is not None:
            return self.base_optimizer

        if self.cost_estimator is None:
            raise ValueError(
                "Chinchilla needs a `cost_estimator` to be given to `autoscan.run()`."
            )
        budget = None
        if budget_info is not None:
            budget = budget_info.total_cost_to_spend
            if budget is None:
                budget = budget_info.worker_cost_to_spend
        if budget is None:
            raise ValueError(
                "Chinchilla needs a cost budget, pass `total_cost_to_spend` or"
                " `worker_cost_to_spend` to `autoscan.run()`."
            )

        self.max_evaluation_flops = budget
        self.min_flops, self.max_flops = self.find_flops_bounds()
        if self.max_cost_per_config is not None:
            self.max_flops = self.max_cost_per_config

        match self.strategy:
            case "approach1":
                config_list, model_configs = self.approach1_generate_config_list()
            case "approach2":
                config_list, model_configs = self.approach2_generate_config_list(
                    num_candidate_per_iso_flop=self.candidates_per_iso_flop
                )

        self.all_possible_model_configs = model_configs
        self.config_list = config_list
        logger.debug(config_list)
        logger.debug(
            "estimated_flops %s evaluations %s max_flops %s",
            sum(self._flops(config) for config in config_list),
            self.max_evaluation_flops,
            self.max_flops,
        )

        self.base_optimizer = GridSearch(configs_list=config_list)
        self.base_optimizer.set_cost_estimator(
            self.cost_estimator, max_cost_per_config=self.max_cost_per_config
        )
        return self.base_optimizer

    def get_all_possible_model_configs(
        self, non_scaling_params: Mapping[str, Any]
    ) -> ModelConfigs:
        parameters = self.space.searchables
        data_param_lower = parameters[self.data_parameter_name].domain.lower
        model_params_grid = {
            param_name: sorted(
                torch.unique(
                    param.domain.cast(
                        torch.linspace(0, 1, steps=30), frm=Domain.unit_float()
                    )
                ).tolist()
            )
            for param_name, param in parameters.items()
            if getattr(param, "is_compute_scaling", False)
            and param_name != self.data_parameter_name
        }
        attempted_model_configs: ModelConfigs = defaultdict(list)
        for indices in itertools.product(
            *[range(len(grid)) for grid in model_params_grid.values()]
        ):
            conf1 = {
                param_name: model_params_grid[param_name][idx]
                for param_name, idx in zip(
                    model_params_grid.keys(), indices, strict=False
                )
            }
            conf1.update(
                {**non_scaling_params, self.data_parameter_name: data_param_lower}
            )
            model_key = tuple(
                sorted((k, v) for k, v in conf1.items() if k != self.data_parameter_name)
            )
            n_params_updated = self._n_params(**conf1)
            attempted_model_configs[n_params_updated].append(model_key)
        return attempted_model_configs

    def find_flops_bounds(self) -> tuple[float, float]:
        searchables = self.space.searchables
        conf_max = {name: param.domain.upper for name, param in searchables.items()}
        conf_min = {name: param.domain.lower for name, param in searchables.items()}
        constants = self.space.constants
        return (
            self._flops({**constants, **conf_min}),
            self._flops({**constants, **conf_max}),
        )

    def approach1_generate_config_list(  # noqa: C901, PLR0912, PLR0915
        self,
    ) -> tuple[list[dict[str, Any]], ModelConfigs]:
        parameters = self.space.searchables
        non_scaling_params = self.sample_non_scaling_parameters()
        data_param_lower = parameters[self.data_parameter_name].domain.lower
        model_params_grid = {
            param_name: sorted(
                torch.unique(
                    param.domain.cast(
                        torch.linspace(0, 1, steps=30), frm=Domain.unit_float()
                    )
                ).tolist()
            )
            for param_name, param in parameters.items()
            if getattr(param, "is_compute_scaling", False)
            and param_name != self.data_parameter_name
        }
        logger.info(f"grid for model params {model_params_grid}")

        param_names = list(model_params_grid.keys())

        current_grid_indices = dict.fromkeys(param_names, 0)

        conf1_scaling = {
            param_name: grid[0] for param_name, grid in model_params_grid.items()
        }

        # Estimate parameters for the base configuration
        base_config = {
            **conf1_scaling,
            **non_scaling_params,
            self.data_parameter_name: data_param_lower,
        }  # data parameter value will be set later
        n_params = self._n_params(**base_config)

        # Get domain bounds and data parameter
        data_param = parameters[self.data_parameter_name]

        # Create dynamic data grid using Chinchilla principle: find data_param values
        # for different multipliers
        assert self.data_multipliers is not None
        data_multipliers = self.data_multipliers

        config_list = []

        for multiplier in data_multipliers:
            data_param_value = int(n_params * multiplier)
            if (
                data_param_value < data_param.domain.lower
                or data_param_value > data_param.domain.upper
            ):
                continue
            conf = {
                **conf1_scaling,
                **non_scaling_params,
                self.data_parameter_name: data_param_value,
            }
            if self.min_flops <= self._flops(conf) <= self.max_flops:
                config_list.append(conf)

        attempted_model_configs: ModelConfigs = defaultdict(list)
        # Add initial config
        if len(config_list) > 0:
            initial_model_key = tuple(
                sorted(
                    (k, v)
                    for k, v in config_list[0].items()
                    if k != self.data_parameter_name
                )
            )
            conf1_n_params = self._n_params(**config_list[0])
            attempted_model_configs[conf1_n_params].append(initial_model_key)
            logger.debug(config_list)

        used_flops = sum([self._flops(conf) for conf in config_list])
        logger.info(f"used flops {used_flops:.2e}")

        action_probabilities = torch.tensor([1.0 / len(param_names)] * len(param_names))
        action_count = 0

        exhausted_dims: set[str] = set()
        while used_flops < self.max_evaluation_flops:
            if len(exhausted_dims) == len(param_names):
                logger.info("All parameters have reached their grid maximums.")
                break

            action = self._choose_action_with_sobol(action_probabilities, action_count)
            action_count += 1

            param_name = param_names[action]
            logger.debug(f"action is {action} parameter to increment {param_name}")

            if param_name in exhausted_dims:
                continue

            current_idx = current_grid_indices[param_name]
            next_idx = current_idx + 1

            if next_idx >= len(model_params_grid[param_name]):
                logger.info(f"Exhausted grid for parameter '{param_name}'.")
                exhausted_dims.add(param_name)
                action_probabilities[action] = 0.0
                if action_probabilities.sum() > 0:
                    action_probabilities = (
                        action_probabilities / action_probabilities.sum()
                    )
                continue

            conf1_scaling = {
                **conf1_scaling,
                param_name: model_params_grid[param_name][next_idx],
            }
            current_config = {**base_config, **conf1_scaling}

            model_key = tuple(
                sorted(
                    (k, v)
                    for k, v in current_config.items()
                    if k != self.data_parameter_name
                )
            )

            # Recompute dynamic data grid for updated scaling parameters
            n_params_updated = self._n_params(**current_config)
            attempted_model_configs[n_params_updated].append(model_key)
            data_param = parameters[self.data_parameter_name]

            # Find data_param values for each multiplier using the updated N
            updated_data_grid = []
            for multiplier in data_multipliers:
                data_param_value = int(n_params_updated * multiplier)
                if (
                    data_param_value < data_param.domain.lower
                    or data_param_value > data_param.domain.upper
                ):
                    continue
                updated_data_grid.append(data_param_value)

            updated_data_grid = sorted(
                set(updated_data_grid)
            )  # Remove duplicates and sort

            for data_grid_value in updated_data_grid:
                current_config[self.data_parameter_name] = data_grid_value
                copy_current = current_config.copy()
                estimated_flops = self._flops(copy_current)
                if (
                    (used_flops + estimated_flops > self.max_evaluation_flops)
                    or (estimated_flops > self.max_flops)
                    or (estimated_flops < self.min_flops)
                ):
                    logger.info(
                        f"Cannot add config {copy_current} due to FLOPs constraint. "
                        f"Used: {used_flops:.2e}, Estimated: {estimated_flops:.2e}, "
                        f"Max: {self.max_evaluation_flops:.2e}"
                    )
                    continue
                config_list.append(copy_current)
                used_flops += estimated_flops
                logger.info(
                    f"Added config {copy_current}. Total used FLOPs: {used_flops:.2e}"
                )

            current_grid_indices[param_name] = next_idx

        # model params + non scaling params (exclude data param)
        logger.info(f"Selected {len(config_list)} unique model configs for approach1")
        attempted_model_configs = self.get_all_possible_model_configs(non_scaling_params)
        return config_list, attempted_model_configs

    def approach2_generate_config_list(  # noqa: C901, PLR0915
        self, num_candidate_per_iso_flop: int = 5
    ) -> tuple[list[dict[str, Any]], ModelConfigs]:

        iso_flops = self._generate_iso_flop_list(num_steps=self.iso_flop_steps)
        logger.info(
            f"Generated {len(iso_flops)} ISO-FLOP values:"
            f" {[f'{f:.2e}' for f in iso_flops]}... (total: {sum(iso_flops):.2e})"
        )

        parameters = self.space.searchables
        non_scaling_params = self.sample_non_scaling_parameters()

        scaling_non_data_grid = {
            param_name: torch.unique(
                param.domain.cast(torch.linspace(0, 1, steps=30), frm=Domain.unit_float())
            ).tolist()
            for param_name, param in parameters.items()
            if getattr(param, "is_compute_scaling", False)
            and param_name != self.data_parameter_name
        }

        data_grid = sorted(
            torch.unique(
                parameters[self.data_parameter_name].domain.cast(
                    torch.linspace(0, 1, steps=100000), frm=Domain.unit_float()
                )
            ).tolist()
        )

        param_names = list(scaling_non_data_grid.keys())
        current_grid_indices = dict.fromkeys(param_names, 0)
        configs_per_iso_flop: defaultdict[float, list[dict[str, Any]]] = defaultdict(list)

        conf_scaling = {
            param_name: grid[0] for param_name, grid in scaling_non_data_grid.items()
        }

        action_probabilities = torch.tensor([1.0 / len(param_names)] * len(param_names))
        action_count = 0
        exhausted_dims: set[str] = set()
        used_flops = 0.0

        selected_model_configs: ModelConfigs = defaultdict(list)
        initial_non_data_conf = {**conf_scaling, **non_scaling_params}
        initial_model_key = tuple(
            sorted((k, v) for k, v in initial_non_data_conf.items())
        )
        initial_param_count = self._n_params(
            **initial_non_data_conf | {self.data_parameter_name: data_grid[0]}
        )
        selected_model_configs[initial_param_count].append(initial_model_key)

        while used_flops < self.max_evaluation_flops and len(exhausted_dims) < len(
            param_names
        ):
            action = self._choose_action_with_sobol(action_probabilities, action_count)
            action_count += 1

            param_name = param_names[action]

            if param_name in exhausted_dims:
                continue

            current_idx = current_grid_indices[param_name]
            next_idx = current_idx + 1

            if next_idx >= len(scaling_non_data_grid[param_name]):
                logger.info(f"Exhausted grid for parameter '{param_name}'.")
                exhausted_dims.add(param_name)
                action_probabilities[action] = 0.0
                if action_probabilities.sum() > 0:
                    action_probabilities = (
                        action_probabilities / action_probabilities.sum()
                    )
                continue

            conf_scaling = conf_scaling.copy()
            conf_scaling[param_name] = scaling_non_data_grid[param_name][next_idx]
            current_grid_indices[param_name] = next_idx

            # Track this model config
            conf_non_data = {**conf_scaling, **non_scaling_params}
            model_key = tuple(sorted((k, v) for k, v in conf_non_data.items()))
            conf_parameter_count = self._n_params(
                **conf_non_data | {self.data_parameter_name: data_grid[0]}
            )
            selected_model_configs[conf_parameter_count].append(model_key)
            replace = False
            param_chosen_iso_flops = []
            for data_val in data_grid:
                config = {
                    **conf_scaling,
                    **non_scaling_params,
                    self.data_parameter_name: data_val,
                }
                conf = config.copy()
                estimated_flops = self._flops(conf)
                if estimated_flops > self.max_flops:
                    logger.info(
                        f"Config {conf} exceeds max FLOPs ({self.max_flops:.2e})."
                        " Skipping."
                    )
                    continue
                available_brackets = [
                    flops
                    for flops in iso_flops
                    if len(configs_per_iso_flop[flops]) <= num_candidate_per_iso_flop
                ]
                if not available_brackets:
                    logger.info(
                        f"No available ISO-FLOP brackets for config {conf} with"
                        f" estimated FLOPs {estimated_flops:.2e}. Skipping."
                    )
                    replace = True
                closest_flop_bin, ratio = self._get_closest_iso_flop(
                    estimated_flops, available_brackets or iso_flops
                )
                if ratio > 1.3 or ratio < 0.7:
                    continue
                if closest_flop_bin in param_chosen_iso_flops:
                    continue
                if replace:
                    logger.info(
                        f"Replacing config in ISO-FLOP bin {closest_flop_bin:.2e}"
                        f" with config {conf} due to lack of available brackets."
                        f" Estimated FLOPs: {estimated_flops:.2e}"
                    )
                    assert configs_per_iso_flop[closest_flop_bin], (
                        f"No configs to replace in ISO-FLOP bin {closest_flop_bin:.2e}"
                    )
                    removed_conf = configs_per_iso_flop[closest_flop_bin][0]
                    removed_flops = self._flops(removed_conf)
                    if (
                        used_flops - removed_flops + estimated_flops
                        <= self.max_evaluation_flops
                    ):
                        configs_per_iso_flop[closest_flop_bin].pop(0)
                        configs_per_iso_flop[closest_flop_bin].append(conf)
                        param_chosen_iso_flops.append(closest_flop_bin)
                        used_flops += estimated_flops - removed_flops
                    continue
                if (used_flops + estimated_flops > self.max_evaluation_flops) or (
                    estimated_flops > self.max_flops
                ):
                    continue
                configs_per_iso_flop[closest_flop_bin].append(conf)
                param_chosen_iso_flops.append(closest_flop_bin)
                used_flops += estimated_flops

        config_list = [
            conf for configs in configs_per_iso_flop.values() for conf in configs
        ]

        logger.info(
            f"Generated {len(config_list)} configs across"
            f" {len(configs_per_iso_flop)} ISO-FLOP levels,"
            f" total FLOPs: {used_flops:.2e} / {self.max_evaluation_flops:.2e}"
        )

        logger.info(f"Selected {len(config_list)} unique model configs for approach2")
        selected_model_configs = self.get_all_possible_model_configs(non_scaling_params)
        return config_list, selected_model_configs

    def _get_closest_iso_flop(
        self, estimated_flops: float, iso_flops: list[float]
    ) -> tuple[float, float]:
        closest = min(iso_flops, key=lambda x: abs(x - estimated_flops))
        return closest, closest / estimated_flops

    def _generate_iso_flop_list(self, num_steps: int) -> list[float]:
        # assume that all the HPs are increasing, the other solution could be to get
        # all possible grids and calc the flops for all and get min and max there.
        return torch.logspace(
            math.log10(self.min_flops * 2), math.log10(self.max_flops), steps=num_steps
        ).tolist()

    def _choose_action_with_sobol(
        self, action_probabilities: torch.Tensor, iteration_count: int
    ) -> int:
        sobol_sampler = Sampler.sobol(ndim=1, scramble=False)

        samples = sobol_sampler.sample(iteration_count + 1, to=Domain.unit_float())
        u = samples[-1].item()

        cumsum = torch.cumsum(action_probabilities, dim=0)
        action = torch.searchsorted(cumsum, torch.tensor(u)).item()

        return min(int(action), len(action_probabilities) - 1)

    def sample_non_scaling_parameters(self) -> dict[str, Any]:
        """The constants and a random value for every parameter that is not scaled."""
        sampled_conf = dict(self.space.constants)
        for param_name, param in self.space.searchables.items():
            if param_name == self.data_parameter_name or getattr(
                param, "is_compute_scaling", False
            ):
                continue
            value = param.domain.cast_one(random.random(), frm=Domain.unit_float())
            if isinstance(param, HPOCategorical):
                sampled_conf[param_name] = param.choices[int(value)]
            else:
                sampled_conf[param_name] = value
        return sampled_conf

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        return self._ensure_grid(budget_info)(trials, budget_info=budget_info, n=n)

    def _evaluated_trials(
        self, trials: Mapping[str, Trial]
    ) -> Iterator[tuple[Trial, _ScalingPoint]]:
        """Every usable evaluation, with its parameter count, data, loss and flops."""
        for trial in trials.values():
            report = trial.report
            if report is None or report.objective_to_minimize is None:
                continue
            loss = report.objective_to_minimize
            if isinstance(loss, Sequence):
                raise NotImplementedError(
                    "Multi-objective Scaling law not implemented yet."
                )
            if not np.isfinite(loss):
                continue

            flops = report.cost
            if flops is None:
                if self.cost_estimator is None:
                    continue
                flops = self.cost_estimator(**trial.config)
            yield (
                trial,
                _ScalingPoint(
                    n_params=self._n_params(**trial.config),
                    n_data=trial.config[self.data_parameter_name],
                    loss=float(loss),
                    flops=flops,
                ),
            )

    def _scaling_points(
        self, trials: Mapping[str, Trial], *, lc_aware: bool
    ) -> list[_ScalingPoint]:
        """One point per evaluated trial, or per point of its learning curve if
        `lc_aware`.
        """
        points: list[_ScalingPoint] = []
        for trial, point in self._evaluated_trials(trials):
            assert trial.report is not None
            curve = (
                _learning_curve_points(trial.report.learning_curve, point.n_params)
                if lc_aware
                else []
            )
            points.extend(curve or [point])
        return points

    def _trials_pareto_front(self, trials: Mapping[str, Trial]) -> list[_ScalingPoint]:
        points = self._scaling_points(trials, lc_aware=self.strategy == "approach1")
        return _pareto_front(points)

    def _plot_flop_vs_param_and_data(
        self, trials: Mapping[str, Trial]
    ) -> tuple[Figure, Figure]:
        front = self._trials_pareto_front(trials)

        xs = [point.flops for point in front]
        ys_params = [float(point.n_params) for point in front]
        ys_data = [float(point.n_data) for point in front]

        fig1 = _create_scaling_plot(xs, ys_params, "Flops", "Params", "FLOPs vs Params")
        fig2 = _create_scaling_plot(xs, ys_data, "Flops", "Data Points", "FLOPs vs Data")

        return fig1, fig2

    def _plot_flops_per_objective_with_lc(
        self, trials: Mapping[str, Trial]
    ) -> Figure | None:
        evaluated = list(self._evaluated_trials(trials))
        if not evaluated:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        # The learning curve of every trial that reported one, sorted by flops
        learning_curves = [
            sorted(curve, key=lambda point: point.flops)
            for trial, point in evaluated
            if (
                curve := _learning_curve_points(
                    trial.report.learning_curve,  # type: ignore[union-attr]
                    point.n_params,
                )
            )
        ]
        param_counts = [point.n_params for _, point in evaluated]
        return self._create_lc_plot(
            param_counts, learning_curves, self._trials_pareto_front(trials)
        )

    def _create_lc_plot(
        self,
        param_counts: list[float],
        learning_curves: list[list[_ScalingPoint]],
        front: list[_ScalingPoint],
    ) -> Figure:
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(10, 6))
        norm = Normalize(vmin=min(param_counts), vmax=max(param_counts))
        cmap = plt.colormaps["viridis"]

        # Plot learning curves, colored by parameter count
        for curve in learning_curves:
            ax.plot(
                [point.flops for point in curve],
                [point.loss for point in curve],
                linewidth=1,
                alpha=0.15,
                color=cmap(norm(curve[0].n_params)),
                zorder=1,
            )

        # Overlay the power law fitted to the Pareto front
        front_flops = [point.flops for point in front]
        fit = fit_power_law(front_flops, [point.loss for point in front])
        if fit is not None:
            ax.plot(
                front_flops,
                _power_law_curve(fit, front_flops),
                "g--",
                linewidth=2.5,
                alpha=0.9,
                label=f"Pareto fit (slope={fit.slope:.3f})",
                zorder=3,
            )
            ax.legend(loc="best", fontsize=10)

        # Configure axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOPs", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.set_title("Training Loss vs FLOPs with Pareto Front (LC-Aware)", fontsize=14)
        ax.grid(visible=True, linestyle="--", alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Parameters")

        plt.tight_layout()
        return fig

    def get_trial_artifacts(
        self, trials: Mapping[str, Trial] | None = None
    ) -> list[Artifact]:
        if trials is None or len(trials) < 5:
            return []

        artifacts: list[Artifact] = []
        try:
            fig_flops_params, fig_flops_data = self._plot_flop_vs_param_and_data(trials)
            artifacts.append(
                Artifact("flops_vs_params", fig_flops_params, ArtifactType.FIGURE)
            )
            artifacts.append(
                Artifact("flops_vs_data", fig_flops_data, ArtifactType.FIGURE)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to generate FLOPs vs Params/Data plots: {e}")

        if self.strategy == "approach1":
            try:
                fig_lc = self._plot_flops_per_objective_with_lc(trials)
                if fig_lc is not None:
                    artifacts.append(
                        Artifact("flops_vs_loss_with_lc", fig_lc, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs vs Loss plot: {e}")
                raise
        return artifacts

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        # Needs no grid, the ids only depend on the trials
        return GridSearch(configs_list=[]).import_trials(
            external_evaluations=external_evaluations,
            trials=trials,
        )
