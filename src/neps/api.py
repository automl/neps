"""API for the neps package.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import ConfigSpace as CS
from typing_extensions import Literal

import metahyper
from metahyper.api import instance_from_map

from .optimizers import BaseOptimizer, SearcherMapping
from .search_spaces.parameter import Parameter
from .search_spaces.search_space import SearchSpace, pipeline_space_from_configspace
from .utils.plotting import (
    Settings,
    get_fig_and_axs,
    map_axs,
    plot_incumbent,
    save_fig,
    set_legend,
)
from .utils.read_results import process_seed
from .utils.result_utils import get_loss


def _debugger_is_active() -> bool:
    """Check if debug mode is active

    Returns:
        bool: is debug mode active
    """
    gettrace = getattr(sys, "gettrace", lambda: None)
    return gettrace() is not None


def _post_evaluation_hook_function(
    _loss_value_on_error: None | float, _ignore_errors: bool
):
    def _post_evaluation_hook(
        config,
        config_id,
        config_working_directory,
        result,
        logger,
        loss_value_on_error=_loss_value_on_error,
        ignore_errors=_ignore_errors,
    ):
        working_directory = Path(config_working_directory, "../../")
        loss = get_loss(result, loss_value_on_error, ignore_errors)

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
    pipeline_space: dict[str, Parameter | CS.ConfigurationSpace]
    | CS.ConfigurationSpace
    | SearchSpace,
    root_directory: str | Path,
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
    ignore_errors: bool = False,
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
        ignore_errors: Ignore hyperparameter settings that threw an error and do not raise
            an error. Error configs still count towards max_evaluations_total.
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
        ValueError: wrong argument values (see the exception message)


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

    logger = logging.getLogger("neps")
    logger.info(f"Starting neps.run using working directory {root_directory}")
    if not isinstance(pipeline_space, SearchSpace):
        try:
            # Support pipeline space as ConfigurationSpace definition
            if isinstance(pipeline_space, CS.ConfigurationSpace):
                pipeline_space = pipeline_space_from_configspace(pipeline_space)

            # Support pipeline space as mix of ConfigurationSpace and neps parameters
            new_pipeline_space: dict[str, Parameter] = dict()
            for key, value in pipeline_space.items():
                if isinstance(value, CS.ConfigurationSpace):
                    config_space_parameters = pipeline_space_from_configspace(value)
                    new_pipeline_space = {**new_pipeline_space, **config_space_parameters}
                else:
                    new_pipeline_space[key] = value
            pipeline_space = new_pipeline_space

            # Transform to neps internal representation of the pipeline space
            pipeline_space = SearchSpace(**pipeline_space)
        except TypeError as e:
            message = f"The pipeline_space has invalid type: {type(pipeline_space)}"
            raise TypeError(message) from e

    if searcher == "default" or searcher is None:
        if pipeline_space.has_fidelity:
            searcher = "hyperband"
            if hasattr(pipeline_space, "has_prior") and pipeline_space.has_prior:
                searcher = "hyperband_custom_default"
        else:
            searcher = "bayesian_optimization"
        logger.info(f"Running {searcher} as the searcher")

    if isinstance(searcher, BaseOptimizer):
        if searcher.budget != budget:
            raise ValueError(
                "Manually initialized searcher with a budget of "
                f"{searcher.budget} instead of {budget}"
            )
        if searcher.pipeline_space is not pipeline_space:
            raise ValueError(
                "Manually initialized searcher with a different pipeline space: "
                f"{searcher.pipeline_space} instead of {pipeline_space}"
            )
        if searcher_kwargs:
            raise ValueError(
                f"Can't pass the args {searcher_kwargs} to an already initialized searcher"
            )
    else:
        searcher_kwargs.update(
            {
                "loss_value_on_error": loss_value_on_error,
                "cost_value_on_error": cost_value_on_error,
                "ignore_errors": ignore_errors,
            }
        )

        searcher = instance_from_map(
            SearcherMapping, searcher, "searcher", as_class=True
        )(
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
        post_evaluation_hook=_post_evaluation_hook_function(
            loss_value_on_error, ignore_errors
        ),
        filesystem_grace_period_for_crashed_configs=0 if _debugger_is_active() else 45,
    )


def plot(
    root_directory: str | Path,
    key_to_extract: str | None = None,
    scientific_mode: bool = False,
    **plotting_kwargs,
) -> None:
    """Plot results of a neural pipeline search run.

    Args:
        root_directory: The directory with neps results (see below).
        scientific_mode:
            - False (default) - root_directory consists of a single run
            - True - root_directory points to tree structure of
                benchmark={}/algorithm={}/seed={}
        key_to_extract: metric to be used on the x-axis (None, "cost", "fidelity")
        **plotting_kwargs: specifies advanced settings for plotting:
            - benchmarks: list of benchmarks to plot
            - algorithms: list of algorithms to plot
            - x_range: tuple (x_min, x_max) specify x-axis bounds
            - log_x: toggles logarithmic scale on the x-axis
            - log_y: toggles logarithmic scale on the y-axis
            - n_workers: in case of parallel runs specify number of parallel processes
            - filename
            - extension: format to save, e.g. "png", "pdf"
            - dpi: resolution of the image

    Raises:
        FileNotFoundError: If the data to be plotted is not present.
    """

    logger = logging.getLogger("neps")
    logger.info(f"Starting neps.plot using working directory {root_directory}")

    settings = Settings(plotting_kwargs)
    logger.info(
        f"Processing {len(settings.benchmarks)} benchmark(s) "
        f"and {len(settings.algorithms)} algorithm(s)..."
    )

    fig, axs = get_fig_and_axs(settings)

    base_path = Path(root_directory)
    output_dir = base_path / "plots"

    for benchmark_idx, benchmark in enumerate(settings.benchmarks):
        if scientific_mode:
            _base_path = os.path.join(base_path, f"benchmark={benchmark}")
            if not os.path.isdir(_base_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), _base_path
                )

        for algorithm in settings.algorithms:
            seeds = [None]
            if scientific_mode:
                _path = os.path.join(_base_path, f"algorithm={algorithm}")
                if not os.path.isdir(_path):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), _path
                    )

                seeds = sorted(os.listdir(_path))  # type: ignore

            incumbents = []
            costs = []
            max_costs = []
            for seed in seeds:
                incumbent, cost, max_cost = process_seed(
                    path=_path if scientific_mode else base_path,
                    seed=seed,
                    algorithm=algorithm,
                    key_to_extract=key_to_extract,
                    n_workers=settings.n_workers,
                )
                incumbents.append(incumbent)
                costs.append(cost)
                max_costs.append(max_cost)

            is_last_row = lambda idx: idx >= (settings.nrows - 1) * settings.ncols
            # pylint: disable=cell-var-from-loop
            is_first_column = lambda idx: benchmark_idx % settings.ncols == 0
            xlabel = "Iterations" if key_to_extract is None else key_to_extract.upper()
            plot_incumbent(
                ax=map_axs(
                    axs,
                    benchmark_idx,
                    len(settings.benchmarks),
                    settings.ncols,
                ),
                x=costs,
                y=incumbents,
                scale_x=max(max_costs) if key_to_extract == "fidelity" else None,
                title=benchmark,
                xlabel=xlabel if is_last_row(benchmark_idx) else None,
                ylabel="Loss" if is_first_column(benchmark_idx) else None,
                log_x=settings.log_x,
                log_y=settings.log_y,
                x_range=settings.x_range,
                label=algorithm,
            )

    set_legend(fig, axs, settings)
    save_fig(fig, output_dir=output_dir, settings=settings)
    logger.info(f"Saved to '{output_dir}/{settings.filename}.{settings.extension}'")
