from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Mapping

import ConfigSpace as CS
import metahyper
import metahyper.old_metahyper.api
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
    pipeline_space: Mapping[str, Parameter] | SearchSpace | CS.ConfigurationSpace,
    working_directory: str | Path,
    n_iterations: int,
    searcher: Literal["bayesian_optimization", "random_search"] = "bayesian_optimization",
    **searcher_kwargs,
):
    if isinstance(pipeline_space, CS.ConfigurationSpace):
        pipeline_space = search_space_from_configspace(pipeline_space)
    elif not isinstance(pipeline_space, SearchSpace):
        pipeline_space = SearchSpace(**pipeline_space)
    else:
        warnings.warn(
            "Using neps.SearchSpace is depreciated and will be removed in the future, please use a dict.",
            DeprecationWarning,
        )

    if searcher == "bayesian_optimization":
        sampler = BayesianOptimization(pipeline_space=pipeline_space, **searcher_kwargs)
    elif searcher == "random_search":
        sampler = RandomSearch(pipeline_space=pipeline_space)  # type: ignore[assignment]
    else:
        raise ValueError

    return metahyper.run(
        run_pipeline,
        sampler,
        working_directory,
        max_evaluations=n_iterations,
    )
