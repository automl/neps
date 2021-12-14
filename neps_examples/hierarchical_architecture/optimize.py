import logging
import time

import neps
from neps_examples.hierarchical_architecture.graph import HierarchicalArchitectureExample


def run_pipeline(  # pylint: disable=unused-argument
    config,
    config_working_directory,
    previous_working_directory,
    target_params: int = 1.5e7,
):
    start = time.time()
    model = config._hyperparameters[  # pylint: disable=W0212
        "graph_grammar"
    ].get_model_for_evaluation()
    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(target_params - number_of_params)
    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "config_id": config.id,
            "val_score": y,
            "test_score": y,
            "train_time": end - start,
        },
    }


if __name__ == "__main__":
    pipeline_space = dict(
        graph_grammar=HierarchicalArchitectureExample(),
    )

    logging.basicConfig(level=logging.INFO)
    result = neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/hierarchical_architecture_example",
        n_iterations=20,
        searcher="bayesian_optimization",
        overwrite_logging=True,
        graph_kernels=["wl"],
    )

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()

    print("Best found configuration:", id2config[incumbent]["config"])
    print("A total of %i unique configurations were sampled." % len(id2config.keys()))
