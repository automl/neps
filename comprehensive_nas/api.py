import logging

import metahyper.api
import numpy as np


class _RandomSampler:
    def __init__(self, config_space):
        self.config_space = config_space
        self.logger = logging.getLogger(__name__)
        self.losses = []
        self.evaluated_configs = []

    def new_result(self, job):
        if job.exception is not None:
            self.logger.warning(f"job {job.id} failed with exception\n{job.exception}")

        self.evaluated_configs.append(job.kwargs["config"])
        self.losses.append(
            job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        )

    def get_config(self):
        return self.config_space.sample_configuration().get_dictionary()


def run_comprehensive_nas(
    run_pipeline,
    config_space,
    working_directory,
    n_iterations,
    searcher="dummy_random",
    start_master=True,
    start_worker=True,
    nic_name="lo",
    do_live_logging=True,
    overwrite_logging=False,
):
    if searcher == "dummy_random":
        sampler = _RandomSampler(config_space=config_space)
    else:
        raise ValueError

    return metahyper.api.run(
        sampler,
        run_pipeline,
        config_space,
        working_directory,
        n_iterations,
        start_master=start_master,
        start_worker=start_worker,
        nic_name=nic_name,
        logger_name="comprehensive_nas",
        do_live_logging=do_live_logging,
        overwrite_logging=overwrite_logging,
    )
