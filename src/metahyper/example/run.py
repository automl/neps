import time
import uuid

import metahyper


class MinimalSampler(metahyper.Sampler):
    def __init__(self):
        super().__init__()
        self.results = dict()

    def load_results(self, results, pending_configs):
        self.results = results

    def get_config_and_ids(self):
        config = dict(a=len(self.results))
        config_id = str(uuid.uuid4())[:6]
        previous_config_id = None
        return config, config_id, previous_config_id


def evaluation_fn(pipeline_directory, **config):  # pylint: disable=unused-argument
    time.sleep(15)
    return 5


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    opt_dir = "test_opt_dir"
    sampler = MinimalSampler()
    metahyper.run(
        evaluation_fn, sampler, optimization_dir=opt_dir, max_evaluations_total=5
    )
    previous_results, pending_configs, _ = metahyper.read(opt_dir)
