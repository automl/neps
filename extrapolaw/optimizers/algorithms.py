"""Extrapolaw Algorithms
===========
The selection of optimization algorithms available in Extrapolaw.

This module conveniently starts with 'a' to be at the top and
is where most of the code documentation for optimizers can be found.

Below you will find some functions with some sane defaults documenting
the parameters available. You can pass these functoins to `extrapolaw.acquire()`
if you like, otherwise you may also refer to them by their string name.
"""

# NOTE: If updating this file with new optimizers, please be aware that
# the documentation here is what is shown in the `extrapolaw.acquire()` documentation.
# Heres a checklist:
# 1. Add you function and document it
# 2. Add it to the `OptimizerChoice` at the bottom of this file.
# 3. Add a section to `extrapolaw.acquire()`

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeAlias

import torch

from extrapolaw.optimizers.bayesian_optimization import BayesianOptimization
from extrapolaw.optimizers.chinchilla import ChinchillaGuidedScaling
from extrapolaw.optimizers.extrapolaw_random_search import (
    ExtrapolawComplexRandomSearch,
    ExtrapolawRandomSearch,
)
from extrapolaw.optimizers.grid_search import GridSearch
from extrapolaw.optimizers.optimizer import AskFunction  # noqa: TC001
from extrapolaw.optimizers.random_search import RandomSearch
from extrapolaw.sampling import Prior, Uniform
from extrapolaw.space.encoding import ConfigEncoder
from extrapolaw.space.extrapolaw_spaces.extrapolaw_space import (
    convert_extrapolaw_to_classic_search_space,
)
from extrapolaw.space.extrapolaw_spaces.parameters import (
    PipelineSpace,
)

if TYPE_CHECKING:
    from extrapolaw.space import SearchSpace


logger = logging.getLogger(__name__)


def _bo(  # noqa: C901, PLR0912
    pipeline_space: SearchSpace | PipelineSpace,
    *,
    initial_design_size: int | Literal["ndim"] = "ndim",
    cost_aware: bool | Literal["log"],
    sample_prior_first: bool,
    ignore_fidelity: bool = False,
    device: torch.device | str | None,
    num_restarts: int = 20,
    n_initial_start_points: int | None = 256,
    acquisition_func_type: Literal["EI", "IT-JES", "IT-MES", "IT-SCAN-JES"] = "EI",
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
    log_y: bool = True,
    budget_schedule: Literal[
        "none", "space_expansion", "left_budget_contraction"
    ] = "none",
    prior_mean: Literal["power_law", "zero"] = "power_law",
    target_flops: int | None = None,
) -> BayesianOptimization:
    """Initialise the BO loop.

    Args:
        pipeline_space: Space in which to search
        initial_design_size: Number of samples used before using the surrogate model.
            If "ndim", it will use the number of parameters in the search space.
        cost_aware: Whether to consider reported "cost" from configurations in decision
            making. If True, the optimizer will weigh potential candidates by how much
            they cost, incentivising the optimizer to explore cheap, good performing
            configurations. This amount is modified over time. If "log", the cost
            will be log-transformed before being used.

            !!! warning

                If using `cost`, cost must be provided in the reports of the trials.

        sample_prior_first: Whether to sample the default configuration first.
        ignore_fidelity: Whether to ignore_fidelity when sampling.
            In this case, the max fidelity is always used.
        device: Device to use for the optimization.
        num_restarts: Number of restarts used when optimizing the acquisition function.
        n_initial_start_points: Number of initial start points used when optimizing the
            acquisition function. If `None`, a dimension-dependent heuristic is used.
        log_y: Whether the GP with the power law prior mean in the cost models the log
            of the objective.
        prior_mean: The prior mean of the GP, the power law in the cost or zero, see
            `BayesianOptimization.prior_mean`.
        budget_schedule: How the max cost of a sampled config follows the spent cost
            budget, see `BayesianOptimization.budget_schedule`.
        target_flops: The FLOPs the `"IT-SCAN-JES"` acquisition samples the optima
            around, which it needs.

    Raises:
        ValueError: if initial_design_size < 1
        ValueError: if fidelity is not None and ignore_fidelity is False
    """
    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is not None:
            pipeline_space = converted_space
        else:
            raise ValueError(
                "This optimizer only supports HPO search spaces, please use an Extrapolaw"
                " space-compatible optimizer."
            )
    if not ignore_fidelity and pipeline_space.fidelity is not None:
        raise ValueError(
            "Fidelities are not supported for BayesianOptimization. Consider setting the"
            " fidelity to a constant value or ignoring it using ignore_fidelity to"
            f" always sample at max fidelity. Got fidelity: {pipeline_space.fidelities} "
        )

    if ignore_fidelity:
        parameters = {**pipeline_space.searchables, **pipeline_space.fidelities}
    else:
        parameters = {**pipeline_space.searchables}

    match initial_design_size:
        case "ndim":
            n_initial_design_size = len(parameters)
        case int():
            if initial_design_size < 1:
                raise ValueError("initial_design_size should be greater than 0")

            n_initial_design_size = initial_design_size
        case _:
            raise ValueError(
                "initial_design_size should be either 'ndim' or a positive integer"
            )

    match device:
        case str():
            device = torch.device(device)
        case None:
            device = torch.get_default_device()
        case torch.device():
            pass
        case _:
            raise ValueError("device should be a string, torch.device or None")

    return BayesianOptimization(
        space=pipeline_space,
        encoder=ConfigEncoder.from_parameters(parameters),
        n_initial_design=n_initial_design_size,
        cost_aware=cost_aware,
        sample_prior_first=sample_prior_first,
        device=device,
        num_restarts=num_restarts,
        n_initial_start_points=n_initial_start_points,
        constraints_func=constraints_func,
        acquisition_func_type=acquisition_func_type,
        log_y=log_y,
        prior_mean=prior_mean,
        budget_schedule=budget_schedule,
        target_flops=target_flops,
    )


def determine_optimizer_automatically(
    space: SearchSpace | PipelineSpace,
) -> tuple[str, dict[str, Any]]:
    """Choose the optimizer for `optimizer="auto"`.

    * A `PipelineSpace` that cannot be converted to a classic search space, for
        example one with `Operation`s, uses `complex_random_search`.
    * A space without priors and fidelities uses `bayesian_optimization`.
    * Otherwise, `random_search` samples from the priors, if there are any.

    A fidelity is always set to its maximum.

    Returns:
        The name of the optimizer and the keyword arguments to create it with.
    """
    if isinstance(space, PipelineSpace):
        has_fidelity = bool(space.fidelity_attrs)
        kwargs: dict[str, Any] = {}
        if has_fidelity:
            kwargs["ignore_fidelity"] = "highest_fidelity"
            _log_pinned_fidelity()
        return "complex_random_search", kwargs

    has_prior = any(
        parameter.prior is not None for parameter in space.searchables.values()
    )
    has_fidelity = len(space.fidelities) > 0
    if not has_prior and not has_fidelity:
        return "bayesian_optimization", {}

    kwargs = {"use_priors": has_prior}
    if has_fidelity:
        kwargs["ignore_fidelity"] = "highest_fidelity"
        _log_pinned_fidelity()
    return "random_search", kwargs


def _log_pinned_fidelity() -> None:
    logger.info(
        "The search space has a fidelity, which the automatically chosen optimizer sets"
        " to its maximum."
    )


def random_search(
    pipeline_space: SearchSpace | PipelineSpace,
    *,
    use_priors: bool = False,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
) -> RandomSearch | ExtrapolawRandomSearch:
    """A simple random search algorithm that samples configurations uniformly at random.

    You may also `use_priors=` to sample from a distribution centered around your defined
    priors.

    Args:
        pipeline_space: The search space to sample from.
        use_priors: Whether to use priors when sampling.
        ignore_fidelity: Whether to ignore_fidelity when sampling.
            Setting this to "highest_fidelity" will always sample at max fidelity.
            Setting this to True will randomly sample from the fidelity like any other
            parameter.
    """
    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is not None:
            pipeline_space = converted_space
        else:
            return extrapolaw_random_search(
                pipeline_space, use_priors=use_priors, ignore_fidelity=ignore_fidelity
            )
    assert ignore_fidelity in (
        True,
        False,
        "highest_fidelity",
    ), "ignore_fidelity should be either True, False or 'highest_fidelity'"
    if not ignore_fidelity and pipeline_space.fidelity is not None:
        raise ValueError(
            "Fidelities are not supported for RandomSearch. Consider setting the"
            " fidelity to a constant value, or setting ignore_fidelity to True to sample"
            " from it like any other parameter or 'highest_fidelity' to always sample at"
            f" max fidelity. Got fidelity: {pipeline_space.fidelities} "
        )
    if ignore_fidelity in (True, "highest_fidelity") and pipeline_space.fidelity is None:
        logger.warning(
            "Warning: You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )
    match ignore_fidelity:
        case True:
            parameters = {**pipeline_space.searchables, **pipeline_space.fidelities}
        case False:
            parameters = {**pipeline_space.searchables}
        case "highest_fidelity":
            parameters = {**pipeline_space.searchables}

    if use_priors and not any(
        parameter.prior is not None for parameter in parameters.values()
    ):
        logger.warning(
            "Warning: You are using priors, but no priors are defined in the search"
            " space. Consider setting use_priors to False."
        )

    if not use_priors and any(
        parameter.prior is not None for parameter in parameters.values()
    ):
        priors = [
            parameter for parameter in parameters.values() if parameter.prior is not None
        ]
        raise ValueError(
            f"To use priors, you must set use_priors=True. Got priors: {priors}"
        )

    return RandomSearch(
        space=pipeline_space,
        encoder=ConfigEncoder.from_parameters(parameters),
        sampler=(
            Prior.from_parameters(parameters)
            if use_priors
            else Uniform(ndim=len(parameters))
        ),
        constraints_func=constraints_func,
    )


def grid_search(
    pipeline_space: SearchSpace | PipelineSpace,
    *,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
    size_per_numerical_dimension: int | dict[str, int] = 5,
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
) -> GridSearch:
    """A simple grid search algorithm which discretizes the search
    space and evaluates all possible configurations.

    Args:
        pipeline_space: The search space to sample from.
        ignore_fidelity: Whether to ignore_fidelity when sampling.
            Setting this to "highest_fidelity" will always sample at max fidelity.
            Setting this to True will make a grid over the fidelity like any other
            parameter.
        size_per_numerical_dimension: The number of points to use per numerical
            dimension when discretizing the space. Can be an integer (applies to all
            numerical hyperparameters) or a dictionary mapping parameter names to
            their grid sizes.
    """
    from extrapolaw.optimizers.utils.grid import make_grid

    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is not None:
            pipeline_space = converted_space
        else:
            return extrapolaw_grid_search(
                pipeline_space,
                ignore_fidelity=ignore_fidelity,
                size_per_numerical_dimension=size_per_numerical_dimension,
                constraints_func=constraints_func,
            )

    if any(
        parameter.prior is not None for parameter in pipeline_space.searchables.values()
    ):
        logger.warning("Grid search does not support priors, they will be ignored.")
    if ignore_fidelity and pipeline_space.fidelity is None:
        logger.warning(
            "Warning: You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )
    if not ignore_fidelity and pipeline_space.fidelity is not None:
        raise ValueError(
            "Fidelities are not supported for GridSearch natively. Consider setting the"
            " fidelity to a constant value, or setting ignore_fidelity to True to sample"
            " from it like any other parameter or 'highest_fidelity' to always sample at"
            f" max fidelity. Got fidelity: {pipeline_space.fidelities} "
        )

    return GridSearch(
        configs_list=make_grid(
            pipeline_space,
            ignore_fidelity=ignore_fidelity,
            size_per_numerical_hp=size_per_numerical_dimension,
        ),
        constraints_func=constraints_func,
    )


def extrapolaw_grid_search(
    pipeline_space: PipelineSpace,
    *,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
    size_per_numerical_dimension: int | dict[str, int] = 5,
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
) -> GridSearch:
    """A simple grid search algorithm which discretizes the search
    space and evaluates all possible configurations.

    Args:
        pipeline_space: The search space to sample from.
        ignore_fidelity: Whether to ignore_fidelity when sampling.
            Setting this to "highest_fidelity" will always sample at max fidelity.
            Setting this to True will make a grid over the fidelity like any other
            parameter.
        size_per_numerical_dimension: The number of points to use per numerical
            dimension when discretizing the space. Can be an integer (applies to all
            numerical hyperparameters) or a dictionary mapping parameter names to
            their grid sizes.
    """
    from extrapolaw.optimizers.utils.grid import make_grid

    if not isinstance(pipeline_space, PipelineSpace):
        raise ValueError(
            "This optimizer only supports Extrapolaw spaces, please use a classic"
            " search space-compatible optimizer."
        )
    if pipeline_space.has_priors():
        logger.warning("Grid search does not support priors, they will be ignored.")
    if not pipeline_space.fidelity_attrs and ignore_fidelity:
        logger.warning(
            "Warning: You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )
    if pipeline_space.fidelity_attrs and not ignore_fidelity:
        raise ValueError(
            "Fidelities are not supported for GridSearch natively. Consider setting the"
            " fidelity to a constant value, or setting ignore_fidelity to True to sample"
            " from it like any other parameter or 'highest_fidelity' to always sample at"
            f" max fidelity. Got fidelity: {pipeline_space.fidelity_attrs} "
        )

    return GridSearch(
        configs_list=make_grid(
            pipeline_space,
            ignore_fidelity=ignore_fidelity,
            size_per_numerical_hp=size_per_numerical_dimension,
        ),
        constraints_func=constraints_func,
    )


def bayesian_optimization(
    pipeline_space: SearchSpace,
    *,
    initial_design_size: int | Literal["ndim"] = "ndim",
    cost_aware: bool | Literal["log"] = False,
    ignore_fidelity: bool = False,
    device: torch.device | str | None = None,
    num_restarts: int = 20,
    n_initial_start_points: int | None = 256,
    constraints_func: Callable[[Mapping[str, Any]], float] | None = None,
    acquisition_func_type: Literal["EI", "IT-JES", "IT-MES", "IT-SCAN-JES"] = "EI",
    log_y: bool = True,
    budget_schedule: Literal[
        "none", "space_expansion", "left_budget_contraction"
    ] = "none",
    prior_mean: Literal["power_law", "zero"] = "power_law",
    target_flops: int | None = None,
    sample_prior_first: bool = False,
) -> BayesianOptimization:
    """Models the relation between hyperparameters in your `pipeline_space`
    and the results of `evaluate_pipeline` using bayesian optimization.
    This acts as a cheap _surrogate model_ of you `evaluate_pipeline` function
    that can be used for optimization.

    !!! tip "When to use this?"

        Bayesion optimization is a good general purpose choice, especially
        if the size of your search space is not too large.

        Note that acquiring the next configuration to evaluate with bayesian
        optimization can become prohibitvely expensive as the number of
        configurations evaluated increases.

    If there is some numeric cost associated with evaluating a configuration,
    you can provide this as a `cost` when returning the results from your
    `evaluate_pipeline` function. By specifying `#!python cost_aware=True`,
    the optimizer will attempt to balance getting the best result while
    minimizing the cost.

    If you have _priors_, use
    [`random_search`][extrapolaw.optimizers.algorithms.random_search] with
    `use_priors=True`, since bayesian optimization does not support them.

    Args:
        pipeline_space: The search space to sample from.
        initial_design_size: Number of samples used before using the surrogate model.
            If "ndim", it will use the number of parameters in the search space.
        cost_aware: Whether to consider reported "cost" from configurations in decision
            making. If True, the optimizer will weigh potential candidates by how much
            they cost, incentivising the optimizer to explore cheap, good performing
            configurations. This amount is modified over time. If "log", the cost
            will be log-transformed before being used.

            !!! warning

                If using `cost`, cost must be provided in the reports of the trials.

        ignore_fidelity: Whether to ignore the fidelity parameter when sampling.
            In this case, the max fidelity is always used.
        device: Device to use for the optimization.

        num_restarts: Number of restarts used when optimizing the acquisition
            function.
        n_initial_start_points: Number of initial start points used when optimizing
            the acquisition function. If `None`, a dimension-dependent heuristic is
            used.
        constraints_func: A function that takes a configuration and returns a float
            indicating how much the configuration violates the constraints. A
            non-negative value means the configuration is valid.
        acquisition_func_type: The acquisition function of single objective
            optimization.
        log_y: Whether the GP with the power law prior mean in the cost models the log
            of the objective.
        prior_mean: The prior mean of the GP, the power law in the cost or zero, see
            `BayesianOptimization.prior_mean`.
        budget_schedule: How the max cost of a sampled config follows the spent cost
            budget, see `BayesianOptimization.budget_schedule`. Needs a
            `cost_estimator` and a cost budget given to `extrapolaw.acquire()`.
        sample_prior_first: Whether the first config is the center of the space.
        target_flops: The FLOPs the `"IT-SCAN-JES"` acquisition samples the optima
            around, which it needs.
    """
    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is not None:
            pipeline_space = converted_space
        else:
            raise ValueError(
                "This optimizer only supports HPO search spaces, please use an Extrapolaw"
                " space-compatible optimizer."
            )

    if not ignore_fidelity and pipeline_space.fidelity is not None:
        raise ValueError(
            "Fidelities are not supported for BayesianOptimization. Consider setting the"
            " fidelity to a constant value or ignoring it using ignore_fidelity to"
            f" always sample at max fidelity. Got fidelity: {pipeline_space.fidelities} "
        )
    if ignore_fidelity and pipeline_space.fidelity is None:
        logger.warning(
            "Warning: You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )

    if any(
        parameter.prior is not None for parameter in pipeline_space.searchables.values()
    ):
        priors = [
            parameter
            for parameter in pipeline_space.searchables.values()
            if parameter.prior is not None
        ]
        raise ValueError(
            "Bayesian optimization does not support priors. Consider using"
            " `random_search` with `use_priors=True` instead."
            f" Got priors: {priors}"
        )

    return _bo(
        pipeline_space=pipeline_space,
        initial_design_size=initial_design_size,
        cost_aware=cost_aware,
        device=device,
        sample_prior_first=sample_prior_first,
        ignore_fidelity=ignore_fidelity,
        num_restarts=num_restarts,
        n_initial_start_points=n_initial_start_points,
        constraints_func=constraints_func,
        acquisition_func_type=acquisition_func_type,
        log_y=log_y,
        prior_mean=prior_mean,
        budget_schedule=budget_schedule,
        target_flops=target_flops,
    )


def carbs(
    pipeline_space: SearchSpace | PipelineSpace,
    *,
    initial_search_radius: float = 0.3,
    num_random_samples: int = 4,
    resample_frequency: int = 0,
    search_center_cost_fraction: float | None = None,
    search_center_shrink_param: str | None = None,
) -> AskFunction:
    """[CARBS](https://github.com/imbue-ai/carbs), a cost-aware Bayesian optimizer that
    searches around the pareto front of objective and cost, starting from a search
    center and moving towards more expensive configurations as it goes.

    !!! tip "When to use this?"

        For scaling studies, where the cost of a configuration varies by orders of
        magnitude and you want good configurations at every cost, not just one.

    CARBS is an external library, install it with `pip install extrapolaw[carbs]`.

    It needs the cost of every evaluation. Either return a `cost` from your
    `evaluate_pipeline`, or pass a `cost_estimator` to `extrapolaw.acquire()`. With
    `max_cost_per_config`, CARBS does not suggest configurations it predicts to be
    more expensive than that.

    ```python
    extrapolaw.acquire(
        ...,
        optimizer="carbs",
        cost_estimator=estimate_flops,
        max_cost_per_config=1e12,
    )
    ```

    !!! note

        Only `Integer` and `Float` parameters are supported, and no fidelities.
        Seeding follows the global seeds, see the reference on seeding.

    Args:
        pipeline_space: The search space to sample from.
        initial_search_radius: How far from a center to sample, as a standard deviation
            in the space CARBS searches in.
        num_random_samples: Sample around the search center until this many
            evaluations succeeded, before the surrogate model is used.
        resample_frequency: Re-evaluate a pareto-optimal configuration every this many
            evaluations, `0` to never do so.
        search_center_cost_fraction: Halve `search_center_shrink_param` until the
            search center costs at most this fraction of `max_cost_per_config`.
        search_center_shrink_param: The parameter to shrink for
            `search_center_cost_fraction`, e.g. the size of the dataset.
    """
    module = _import_external_optimizer("extrapolaw.optimizers.carbs", extra="carbs")

    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is None:
            raise ValueError(
                "CARBS only supports HPO search spaces, please use an Extrapolaw"
                " space-compatible optimizer."
            )
        pipeline_space = converted_space

    return module.CARBSOptimizer(
        space=pipeline_space,
        initial_search_radius=initial_search_radius,
        num_random_samples=num_random_samples,
        resample_frequency=resample_frequency,
        search_center_cost_fraction=search_center_cost_fraction,
        search_center_shrink_param=search_center_shrink_param,
    )


def chinchilla(
    pipeline_space: SearchSpace | PipelineSpace,
    *,
    data_parameter_name: str,
    params_estimator: Callable[..., float],
    strategy: Literal["approach1", "approach2"],
    data_multipliers: Sequence[float] | None = None,
    iso_flop_steps: int = 10,
    candidates_per_iso_flop: int = 10,
) -> ChinchillaGuidedScaling:
    """Chinchilla guided scaling, following
    [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
    Lays out a list of configurations that grow the model and the amount of data
    together, to fit how the two should scale with the compute.

    !!! tip "When to use this?"

        For scaling studies, where you want to know how to split a compute budget
        between the size of the model and the amount of data it is trained on.

    The parameters marked with `is_compute_scaling=True` make up the model, the one
    named `data_parameter_name` sets the amount of data. Every other parameter is
    drawn at random once and held fixed for the whole run.

    It needs a `cost_estimator` and a cost budget to be given to `extrapolaw.acquire()`.
    With `max_cost_per_config`, no configuration estimated to be more expensive than
    that is laid out.

    ```python
    extrapolaw.acquire(
        ...,
        optimizer=(
            "chinchilla",
            {"data_parameter_name": "n_data", "params_estimator": estimate_params},
        ),
        cost_estimator=estimate_flops,
        max_cost_per_config=1e12,
        total_cost_to_spend=1e14,
    )
    ```

    !!! note

        Fidelities are not supported, and the configurations are evaluated one at a
        time per worker.

    Args:
        pipeline_space: The search space to sample from.
        data_parameter_name: The parameter that sets the amount of training data.
        params_estimator: Called as `params_estimator(**config)`, returns the
            parameter count of the model.
        strategy: How to lay out the configurations.

            * `"approach1"`: Train every model on `data_multipliers` times its
                parameter count.
            * `"approach2"`: Place the configurations on ISO-FLOP levels, spaced
                logarithmically up to `max_cost_per_config`.
        data_multipliers: The data to parameter count ratios of `"approach1"`.
        iso_flop_steps: The number of ISO-FLOP levels of `"approach2"`.
        candidates_per_iso_flop: How many configurations to place on an ISO-FLOP
            level.
    """
    if isinstance(pipeline_space, PipelineSpace):
        converted_space = convert_extrapolaw_to_classic_search_space(pipeline_space)
        if converted_space is None:
            raise ValueError(
                "Chinchilla only supports HPO search spaces, please use an Extrapolaw"
                " space-compatible optimizer."
            )
        pipeline_space = converted_space

    return ChinchillaGuidedScaling(
        space=pipeline_space,
        data_parameter_name=data_parameter_name,
        params_estimator=params_estimator,
        strategy=strategy,
        data_multipliers=data_multipliers,
        iso_flop_steps=iso_flop_steps,
        candidates_per_iso_flop=candidates_per_iso_flop,
    )


def _import_external_optimizer(module: str, *, extra: str) -> Any:
    """Import the module of an optimizer that wraps an optional, external library."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"The '{extra}' optimizer needs an external library that is not installed."
            f" Install it with `pip install extrapolaw[{extra}]`."
        ) from e


@dataclass
class CustomOptimizer:
    """Custom optimizer that allows you to define your own optimizer function.

    Args:
        optimizer: The optimizer function to use.
    """

    name: str
    optimizer: Callable[Concatenate[SearchSpace, ...], AskFunction] | AskFunction
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    initialized: bool = False

    def create(self, space: SearchSpace | PipelineSpace) -> AskFunction:
        assert not self.initialized, "Custom optimizer already initialized."
        return self.optimizer(space, **self.kwargs)  # type: ignore


def custom(
    name: str,
    optimizer: Callable[Concatenate[SearchSpace, ...], AskFunction] | AskFunction,
    *,
    initialized: bool = False,
    kwargs: Mapping[str, Any] | None = None,
) -> CustomOptimizer:
    """Create a custom optimizer that allows you to define your own optimizer function.

    Args:
        name: The name of the optimizer.
        optimizer: The optimizer function to use.
        initialized: Whether the optimizer has already been initialized.
        **kwargs: Additional arguments to pass to the optimizer function.
    """
    return CustomOptimizer(
        name=name,
        optimizer=optimizer,
        kwargs=kwargs or {},
        initialized=initialized,
    )


def complex_random_search(
    pipeline_space: PipelineSpace,
    *,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
) -> ExtrapolawComplexRandomSearch:
    """A complex random search algorithm that samples configurations uniformly at random,
    but allows for more complex sampling strategies.

    Args:
        pipeline_space: The search space to sample from.
        ignore_fidelity: Whether to ignore the fidelity parameter when sampling.
            If `True`, the algorithm will sample the fidelity like a normal parameter.
            If set to `"highest_fidelity"`, it will always sample at the highest_fidelity.
    Raises:
        ValueError: If the pipeline has fidelity attributes and `ignore_fidelity` is
            set to `False`. Complex random search does not support fidelities by default.
    """

    if pipeline_space.fidelity_attrs and ignore_fidelity is False:
        raise ValueError(
            "Complex Random Search does not support fidelities by default."
            "Consider using `ignore_fidelity=True` or `highest_fidelity`"
            "to always sample at max fidelity."
        )
    if not pipeline_space.fidelity_attrs and ignore_fidelity is not False:
        logger.warning(
            "You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )

    return ExtrapolawComplexRandomSearch(
        pipeline=pipeline_space,
        ignore_fidelity=ignore_fidelity,
    )


def extrapolaw_random_search(
    pipeline_space: PipelineSpace,
    *,
    use_priors: bool = False,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
) -> ExtrapolawRandomSearch:
    """A simple random search algorithm that samples configurations uniformly at random.

    Args:
        pipeline_space: The search space to sample from.
        use_priors: Whether to use priors when sampling.
            If `True`, the algorithm will sample from the prior distribution
            defined in the search space.
        ignore_fidelity: Whether to ignore the fidelity parameter when sampling.
            If `True`, the algorithm will sample the fidelity like a normal parameter.
            If set to `"highest_fidelity"`, it will always sample at the highest_fidelity.
    Raises:
        ValueError: If the pipeline space has fidelity attributes and `ignore_fidelity` is
            set to `False`. Random search does not support fidelities by default.
    """

    if pipeline_space.fidelity_attrs and ignore_fidelity is False:
        raise ValueError(
            "Random Search does not support fidelities by default."
            "Consider using `ignore_fidelity=True` or `highest_fidelity`"
            "to always sample at max fidelity."
        )
    if not pipeline_space.fidelity_attrs and ignore_fidelity is not False:
        logger.warning(
            "You are using ignore_fidelity, but no fidelity is defined in the"
            " search space. Consider setting ignore_fidelity to False."
        )
    if use_priors and not pipeline_space.has_priors():
        logger.warning(
            "You have set use_priors=True, but no priors are defined in the search space."
        )

    return ExtrapolawRandomSearch(
        pipeline=pipeline_space, use_priors=use_priors, ignore_fidelity=ignore_fidelity
    )


PredefinedOptimizers: Mapping[str, Any] = {
    f.__name__: f
    for f in (
        bayesian_optimization,
        random_search,
        grid_search,
        extrapolaw_random_search,
        complex_random_search,
        carbs,
        chinchilla,
    )
}

OptimizerChoice: TypeAlias = Literal[
    "bayesian_optimization",
    "random_search",
    "grid_search",
    "extrapolaw_random_search",
    "complex_random_search",
    "carbs",
    "chinchilla",
]
