"""API for the neps package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping

import ConfigSpace as CS
import metahyper
from typing_extensions import Literal

from .search_spaces.parameter import Parameter

try:
    import torch as _  # Not needed in api.py, but test if torch can be imported
except ModuleNotFoundError:
    from .utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message) from None

from .optimizers.bayesian_optimization.optimizer import BayesianOptimization
from .optimizers.random_search.optimizer import RandomSearch
from .search_spaces.search_space import SearchSpace, pipeline_space_from_configspace


def _post_evaluation_hook(config, config_id, config_working_directory, result, logger):
    working_directory = Path(config_working_directory, "../../")

    is_error = result == "error"
    if is_error:
        loss = float("inf")
    elif isinstance(result, dict):
        loss = result["loss"]
    else:
        loss = result

    # 1. write all configs and losses
    all_configs_losses = Path(working_directory, "all_losses_and_configs.txt")

    def write_loss_and_config(file_handle, loss_, config_id_, config_):
        file_handle.write(f"Loss: {loss_}\n")
        file_handle.write(f"Config ID: {config_id_}\n")
        file_handle.write(f"Config: {config_}\n")
        file_handle.write(79 * "-" + "\n")

    with all_configs_losses.open("a", encoding="utf-8") as f:
        write_loss_and_config(f, loss, config_id, config)

    # No need to handle best loss cases if an error occurred
    if is_error:
        return

    # The "best" loss exists only in the pareto sense for multi-objective
    is_multi_objective = isinstance(loss, dict)
    if is_multi_objective:
        logger.info(f"Finished evaluating config {config_id}")
        return

    # 2. Write best losses / configs
    best_loss_trajectory_file = Path(working_directory, "best_loss_trajectory.txt")
    best_loss_config_trajectory_file = Path(
        working_directory, "best_loss_with_config_trajectory.txt"
    )

    if not best_loss_trajectory_file.exists():
        is_new_best = result != "error"
    else:
        best_loss_trajectory = best_loss_trajectory_file.read_text(encoding="utf-8")
        best_loss_trajectory = list(best_loss_trajectory.rstrip("\n").split("\n"))
        best_loss = best_loss_trajectory[-1]
        is_new_best = float(best_loss) > loss

    if is_new_best:
        with best_loss_trajectory_file.open("a", encoding="utf-8") as f:
            f.write(f"{loss}\n")

        with best_loss_config_trajectory_file.open("a", encoding="utf-8") as f:
            write_loss_and_config(f, loss, config_id, config)

        logger.info(
            f"Finished evaluating config {config_id}"
            f" -- new best with loss {loss :.3f}"
        )
    else:
        logger.info(f"Finished evaluating config {config_id}")


def run(
    run_pipeline: Callable,
    pipeline_space: dict[str, Parameter | CS.ConfigurationSpace] | CS.ConfigurationSpace,
    working_directory: str | Path,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    searcher: Literal["bayesian_optimization", "random_search"] = "bayesian_optimization",
    run_pipeline_args: Iterable | None = None,
    run_pipeline_kwargs: Mapping | None = None,
    **searcher_kwargs,
) -> None:
    """Run a neural pipeline search.

    To parallelize:
        Simply call run(.) multiple times (optionally on different machines). Make sure
        that working_directory points to the same folder in the same filesystem, otherwise
        the multiple calls to run(.) will be independent.

    Args:
        run_pipeline: The objective function to minimize.
        pipeline_space: The search space to minimize over.
        working_directory: The directory to save progress to. This is also used to
            synchronize multiple calls to run(.) for parallelization.
        max_evaluations_total: Number of evaluations after which to terminate.
        max_evaluations_per_run: Number of evaluations the specific call to run(.) should
            maximally do.
        continue_until_max_evaluation_completed: If true, only stop after
            max_evaluations_total have been completed. This is only relevant in the
            parallel setting.
        searcher: Which optimizer to use.
        run_pipeline_args: Positional arguments that are passed to run_pipeline.
        run_pipeline_kwargs: Keywoard arguments that are passed to run_pipeline.
        **searcher_kwargs: Will be passed to the searcher. This is usually only needed by
            neps develolpers.

    Raises:
        TypeError: If pipeline_space has invalid type.
        ValueError: If searcher is unkown.

    Example:
        >>> import neps

        >>> def run_pipeline(some_parameter: float):
        >>>    validation_error = -some_parameter
        >>>    return validation_error

        >>> pipeline_space = dict(some_parameter=neps.FloatParameter(lower=0, upper=1))

        >>> neps.run(
        >>>    run_pipeline=run_pipeline,
        >>>    pipeline_space=pipeline_space,
        >>>    working_directory="usage_example",
        >>>    max_evaluations_total=5,
        >>> )
    """
    try:
        # Support pipeline space as ConfigurationSpace definition
        if isinstance(pipeline_space, CS.ConfigurationSpace):
            pipeline_space = pipeline_space_from_configspace(pipeline_space)

        # Support pipeline space as mix of ConfigurationSpace and neps parameters
        for key, value in pipeline_space.items():
            if isinstance(value, CS.ConfigurationSpace):
                pipeline_space.pop(key)
                config_space_parameters = pipeline_space_from_configspace(value)
                pipeline_space = {**pipeline_space, **config_space_parameters}

        # Transform to neps internal representation of the pipeline space
        pipeline_space = SearchSpace(**pipeline_space)
    except TypeError as e:
        message = f"The pipeline_space has invalid type: {type(pipeline_space)}"
        raise TypeError(message) from e

    if searcher == "bayesian_optimization":
        sampler = BayesianOptimization(pipeline_space=pipeline_space, **searcher_kwargs)
    elif searcher == "random_search":
        sampler = RandomSearch(pipeline_space=pipeline_space)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown searcher: {searcher}")

    metahyper.run(
        run_pipeline,
        sampler,
        working_directory,
        max_evaluations_total=max_evaluations_total,
        max_evaluations_per_run=max_evaluations_per_run,
        continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
        logger=logging.getLogger("neps"),
        evaluation_fn_args=run_pipeline_args,
        evaluation_fn_kwargs=run_pipeline_kwargs,
        post_evaluation_hook=_post_evaluation_hook,
    )
