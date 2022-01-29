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
        raise ValueError

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
    )
