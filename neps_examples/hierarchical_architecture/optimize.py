import logging
import time

import neps
from neps_examples.hierarchical_architecture.graph import HierarchicalArchitectureExample


def run_pipeline(
    working_directory,
    architecture,
    target_params: float = 1.5e7,
):
    start = time.time()
    model = architecture.get_model_for_evaluation()
    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(target_params - number_of_params)
    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


pipeline_space = dict(
    architecture=HierarchicalArchitectureExample(),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/hierarchical_architecture_example",
    max_evaluations_total=20,
)

previous_results, pending_configs = neps.status(
    "results/hierarchical_architecture_example"
)
