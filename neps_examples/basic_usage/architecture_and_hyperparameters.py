import logging

import neps


def run_pipeline(**config):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    model = config["graph"].to_pytorch()

    target_params = 1531258
    number_of_params = sum(p.numel() for p in model.parameters())
    validation_error = abs(target_params - number_of_params) / target_params

    target_lr = 10e-3
    validation_error += abs(target_lr - learning_rate) / target_lr
    validation_error += int(optimizer == "sgd")

    return validation_error


nb201_choices = [
    "ReLUConvBN3x3",
    "ReLUConvBN1x1",
    "AvgPool1x1",
    "Identity",
    "Zero",
]

pipeline_space = dict(
    graph=neps.GraphDenseParameter(
        num_nodes=4, edge_choices=nb201_choices, in_channels=3, num_classes=10
    ),
    optimizer=neps.CategoricalParameter(choices=["sgd", "adam"]),
    learning_rate=neps.FloatParameter(lower=10e-7, upper=10e-3, log=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_architecture_example",
    max_evaluations_total=15,
)
