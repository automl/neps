import metahyper.api

from .optimizers.bayesian_optimization.optimizer import BayesianOptimization
from .optimizers.random_search.new_optimizer_dummy import _DummySearcher


def run_comprehensive_nas(
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
    if searcher == "dummy_random":
        sampler = _DummySearcher(pipeline_space=pipeline_space)
    elif searcher == "bayesian_optimization":
        sampler = BayesianOptimization(pipeline_space=pipeline_space, **searcher_kwargs)
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
        logger_name="comprehensive_nas",
        do_live_logging=do_live_logging,
        overwrite_logging=overwrite_logging,
    )
