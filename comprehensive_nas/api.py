import metahyper.api

from comprehensive_nas.optimizers.random_search.new_optimizer_dummy import _DummySearcher


class PipelineSpace:
    def __init__(self, **parameters_kwargs):
        self.parameters = parameters_kwargs


class _Parameter:
    pass


class Categorical(_Parameter):
    def __init__(self, choices):
        pass


class Float(_Parameter):
    def __init__(self, lower, upper, log=False):
        pass


class Integer(_Parameter):
    def __init__(self, lower, upper, log=False):
        pass


class GrammarGraph(_Parameter):
    def __init__(self):
        pass


class DenseGraph(_Parameter):
    def __init__(self, num_nodes, edge_choices):
        pass


def run_comprehensive_nas(
    run_pipeline,
    pipeline_space,
    working_directory,
    n_iterations,
    searcher="dummy_random",
    start_master=True,
    start_worker=True,
    nic_name="lo",
    do_live_logging=True,
    overwrite_logging=False,
    **searcher_kwargs,  # pylint: disable=unused-argument
):
    if searcher == "dummy_random":
        sampler = _DummySearcher(pipeline_space=pipeline_space)
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
