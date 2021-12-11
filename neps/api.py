import metahyper.api

try:
    import torch as _  # Not needed in api.py, but test if torch can be imported
except ModuleNotFoundError:
    from neps.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message) from None

from .optimizers.bayesian_optimization.optimizer import BayesianOptimization
from .optimizers.random_search.optimizer import RandomSearch


def run(
    run_pipeline,
    pipeline_space,
    working_directory,
    n_iterations,
    searcher="bayesian_optimization",  # TODO: Naming uniform
    start_master=True,
    start_worker=True,
    nic_name="lo",
    do_live_logging=False,
    overwrite_logging=False,
    **searcher_kwargs,  # pylint: disable=unused-argument
):
    if searcher == "bayesian_optimization":
        sampler = BayesianOptimization(pipeline_space=pipeline_space, **searcher_kwargs)
    elif searcher == "random_search":
        sampler = RandomSearch(pipeline_space=pipeline_space)
    else:
        raise ValueError

    return metahyper.api.run(
        sampler,
        run_pipeline,
        pipeline_space,
        working_directory,
        n_iterations,
        start_master=start_master,
        start_worker=start_worker,
        nic_name=nic_name,
        logger_name="neps",
        do_live_logging=do_live_logging,
        overwrite_logging=overwrite_logging,
    )
