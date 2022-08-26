"""API for the neps package.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable

import ConfigSpace as CS
import metahyper
from metahyper.api import instance_from_map
from typing_extensions import Literal

from .search_spaces.parameter import Parameter
from .utils.result_utils import get_loss

try:
    import torch as _  # Not needed in api.py, but test if torch can be imported
except ModuleNotFoundError:
    from .utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message) from None

import warnings

from .optimizers import BaseOptimizer, SearcherMapping
from .search_spaces.search_space import SearchSpace, pipeline_space_from_configspace

warnings.simplefilter("always", DeprecationWarning)


def _post_evaluation_hook_function(_loss_value_on_error: None | float):
    def _post_evaluation_hook(
        config,
        config_id,
        config_working_directory,
        result,
        logger,
        loss_value_on_error=_loss_value_on_error,
    ):
        working_directory = Path(config_working_directory, "../../")
        # TODO (Nils): Should loss_value_on_error be passed here as well or continue with
        #  the default inf?
        loss = get_loss(result, loss_value_on_error)

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
        if result == "error":
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
                f" -- new best with loss {float(loss) :.3f}"
            )
        else:
            logger.info(f"Finished evaluating config {config_id}")

    return _post_evaluation_hook


def run(
    run_pipeline: Callable,
    pipeline_space: dict[str, Parameter | CS.ConfigurationSpace] | CS.ConfigurationSpace,
    root_directory: None | str | Path = None,
    overwrite_working_directory: bool = False,
    development_stage_id=None,
    task_id=None,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    budget: int | float | None = None,
    continue_until_max_evaluation_completed: bool = False,
    searcher: Literal[
        "default",
        "bayesian_optimization",
        "random_search",
        "cost_cooling",
        "mf_bayesian_optimization",
        "grid_search",
    ]
    | BaseOptimizer = "default",
    serializer: Literal["yaml", "dill", "json"] = "yaml",
    loss_value_on_error: None | float = None,
    cost_value_on_error: None | float = None,
    **searcher_kwargs,
) -> None:
    """Run a neural pipeline search.

    To parallelize:
        In order to run a neural pipeline search with multiple processes or machines,
        simply call run(.) multiple times (optionally on different machines). Make sure
        that working_directory points to the same folder on the same filesystem, otherwise
        the multiple calls to run(.) will be independent.

    Args:
        run_pipeline: The objective function to minimize.
        pipeline_space: The search space to minimize over.
        root_directory: The directory to save progress to. This is also used to
            synchronize multiple calls to run(.) for parallelization.
        overwrite_working_directory: If true, delete the working directory at the start of
            the run.
        development_stage_id: ID for the current development stage. Only needed if
            you work with multiple development stages.
        task_id: ID for the current task. Only needed if you work with multiple
            tasks.
        max_evaluations_total: Number of evaluations after which to terminate.
        max_evaluations_per_run: Number of evaluations the specific call to run(.) should
            maximally do.
        budget: Maximum allowed budget. Currently, can be exceeded, but no new evaluations
            will start when the budget it depleted.
        continue_until_max_evaluation_completed: If true, only stop after
            max_evaluations_total have been completed. This is only relevant in the
            parallel setting.
        searcher: Which optimizer to use.
        serializer: Serializer to store hyperparameters configurations. Can be an object,
            or a value in 'json', 'yaml' or 'dill' (see metahyper).
        loss_value_on_error: Setting this and cost_value_on_error to any float will
            supress any error and will use given loss value instead. default: None
        cost_value_on_error: Setting this and loss_value_on_error to any float will
            supress any error and will use given cost value instead. default: None
        **searcher_kwargs: Will be passed to the searcher. This is usually only needed by
            neps develolpers.

    Raises:
        ValueError: If deprecated argument working_directory is used.
        ValueError: If root_directory is None.
        TypeError: If pipeline_space has invalid type.


    Example:
        >>> import neps

        >>> def run_pipeline(some_parameter: float):
        >>>    validation_error = -some_parameter
        >>>    return validation_error

        >>> pipeline_space = dict(some_parameter=neps.FloatParameter(lower=0, upper=1))

        >>> logging.basicConfig(level=logging.INFO)
        >>> neps.run(
        >>>    run_pipeline=run_pipeline,
        >>>    pipeline_space=pipeline_space,
        >>>    root_directory="usage_example",
        >>>    max_evaluations_total=5,
        >>> )
    """
    if "working_directory" in searcher_kwargs:
        raise ValueError(
            "The argument 'working_directory' is deprecated, please use 'root_directory' instead"
        )

    if root_directory is None:
        raise ValueError("'root_directory' can not be None")

    logger = logging.getLogger("neps")
    logger.info(f"Starting neps.run using working directory {root_directory}")
    try:
        # Support pipeline space as ConfigurationSpace definition
        if isinstance(pipeline_space, CS.ConfigurationSpace):
            pipeline_space = pipeline_space_from_configspace(pipeline_space)

        # Support pipeline space as mix of ConfigurationSpace and neps parameters
        for key, value in copy.copy(pipeline_space).items():
            if isinstance(value, CS.ConfigurationSpace):
                pipeline_space.pop(key)
                config_space_parameters = pipeline_space_from_configspace(value)
                pipeline_space = {**pipeline_space, **config_space_parameters}

        # Transform to neps internal representation of the pipeline space
        pipeline_space = SearchSpace(**pipeline_space)
    except TypeError as e:
        message = f"The pipeline_space has invalid type: {type(pipeline_space)}"
        raise TypeError(message) from e

    if searcher == "default" or searcher is None:
        if pipeline_space.has_fidelity:
            searcher = "mf_bayesian_optimization"
        else:
            searcher = "bayesian_optimization"

    searcher_kwargs.update(
        {
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
        }
    )
    searcher = instance_from_map(SearcherMapping, searcher, "searcher", as_class=True)(
        pipeline_space=pipeline_space,
        budget=budget,
        **searcher_kwargs,
    )

    metahyper.run(
        run_pipeline,
        searcher,
        root_directory,
        development_stage_id=development_stage_id,
        task_id=task_id,
        max_evaluations_total=max_evaluations_total,
        max_evaluations_per_run=max_evaluations_per_run,
        overwrite_optimization_dir=overwrite_working_directory,
        continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
        serializer=serializer,
        logger=logger,
        post_evaluation_hook=_post_evaluation_hook_function(loss_value_on_error),
    )
