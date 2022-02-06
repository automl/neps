"""API for the neps package.
"""

from __future__ import annotations

import logging
import warnings
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
from .search_spaces.search_space import SearchSpace, search_space_from_configspace


def _post_evaluation_hook(config, config_id, config_working_directory, result, logger):
    best_loss_trajectory_file = Path(
        config_working_directory, "../../best_loss_trajectory.txt"
    )

    loss = result["loss"]
    if isinstance(loss, dict):
        logger.info(f"Finished evaluating config {config_id}")
        return  # The post evaluation hook does not support multiple objectives yet.

    if not best_loss_trajectory_file.exists():
        is_new_best = True
    else:
        best_loss_trajectory = best_loss_trajectory_file.read_text(encoding="utf-8")
        best_loss_trajectory = list(best_loss_trajectory.rstrip("\n").split("\n"))
        best_loss = best_loss_trajectory[-1]
        is_new_best = float(best_loss) > loss

    if is_new_best:
        with best_loss_trajectory_file.open("a", encoding="utf-8") as f:
            f.write(f"{loss}\n")

        logger.info(
            f"Finished evaluating config {config_id}"
            f" -- new best with loss {loss :.3f}"
        )
    else:
        logger.info(f"Finished evaluating config {config_id}")


def run(
    run_pipeline: Callable,
    pipeline_space: Mapping[str, Parameter] | CS.ConfigurationSpace,
    working_directory: str | Path,
    n_iterations: int | None = None,
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
        n_iterations: deprecated!
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

        >>> def run_pipeline(x):
        >>>    return {"loss": x}

        >>> pipeline_space = dict(x=neps.FloatParameter(lower=0, upper=1, log=False))

        >>> neps.run(
        >>>    run_pipeline=run_pipeline,
        >>>    pipeline_space=pipeline_space,
        >>>    working_directory="results/usage",
        >>>    max_evaluations_total=5,
        >>>    hp_kernels=["m52"],
        >>> )
    """
    if isinstance(pipeline_space, CS.ConfigurationSpace):
        pipeline_space = search_space_from_configspace(pipeline_space)
    else:
        try:
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

    if n_iterations is not None:
        warnings.warn(
            "n_iterations is deprecated and will be removed in a future version",
            DeprecationWarning,
        )
        max_evaluations_total = n_iterations
        continue_until_max_evaluation_completed = True  # Old behavior

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
