import logging
import time

import neps as cnas


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    config_dict = config._hyperparameters  # pylint: disable=protected-access

    # optimizer = config_dict["optimizer"].value  # pylint: disable=unused-argument
    learning_rate = config_dict["learning_rate"].value
    model = config_dict["graph"].get_model_for_evaluation()

    start = time.time()

    number_of_params = sum(p.numel() for p in model.parameters())
    y = number_of_params

    y += 100000 * learning_rate

    end = time.time()

    return {
        "loss": -y,
        "info_dict": {
            "config_id": config.id,
            "val_score": -y,
            "test_score": -y,
            "train_time": end - start,
        },
    }


if __name__ == "__main__":

    nb201_choices = [
        "ReLUConvBN3x3",
        "ReLUConvBN1x1",
        "AvgPool1x1",
        "Identity",
        "Zero",
    ]

    pipeline_space = cnas.SearchSpace(
        graph=cnas.GraphDenseParameter(num_nodes=4, edge_choices=nb201_choices),
        optimizer=cnas.CategoricalParameter(choices=["sgd", "adam"]),
        learning_rate=cnas.FloatParameter(lower=10e-7, upper=10e-3, log=True),
    )

    logging.basicConfig(level=logging.INFO)
    result = cnas.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/hyperparameters_architecture_example",
        n_iterations=50,
        searcher="bayesian_optimization",
        overwrite_logging=True,
        hp_kernels=["m52", "hm"],
        graph_kernels=["wl"],
    )

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()

    print("Best found configuration:", id2config[incumbent]["config"])
    print("A total of %i unique configurations were sampled." % len(id2config.keys()))
