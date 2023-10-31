"""
This code is not runnable but should serve as a guide to a successful neps run
using priorband as a searcher.

Steps:
1. Create search space with a fidelity parameter.
2. Create run_pipeline which includes:
    a. Load the checkpoints if they exist from previous_pipeline_directory.
    b. Train or continue training the model.
    c. Save the model in the new checkpoint which should be located in
        pipeline_directory (current).
    d. Return the loss or the info dictionary.
3. Use neps.run and specify "priorband" as the searcher.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import neps
from neps.utils.common import load_checkpoint, save_checkpoint


class my_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=392)
        self.linear2 = nn.Linear(in_features=392, out_features=196)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


def pipeline_space() -> dict:
    # Create the search space based on NEPS parameters and return the dictionary.
    # IMPORTANT: The search space should have default values for all parameters
    #   which will be used as priors in the priorband search.
    space = dict(
        weight_decay=neps.FloatParameter(lower=1e-5, upper=1e-2, default=5e-4, log=True),
        lr=neps.FloatParameter(lower=1e-5, upper=1e-2, default=1e-3, log=True),
        optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"], default="Adam"),
        epochs=neps.IntegerParameter(lower=2, upper=10, is_fidelity=True),
    )
    return space


# NOTE: The order of the arguments in the run_pipeline function is important.
def run_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict:
    # 1. Create your model and the optimizer according to the coniguration
    model = my_model()

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError(
            "Optimizer choices are defined differently in the pipeline_space"
        )

    # 2. Load the checkpoints if they exist from previous_pipeline_directory
    loaded_values = load_checkpoint(
        previous_pipeline_directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer,
    )

    if loaded_values is not None:
        epoch_already_trained = loaded_values["epochs"]
        # + Anything else saved in the checkpoint.
    else:
        epoch_already_trained = 0
        # + Anything else with default value.

    # 3. Train or continue training the model based on the specified checkpoint
    max_epochs = config["epochs"]
    for epoch in range(epoch_already_trained, max_epochs):
        val_loss = 0
        print(f"The validation loss at epoch {epoch} is {val_loss}")

    # 4. Save the checkpoint data in the current directory
    save_checkpoint(
        pipeline_directory=pipeline_directory,
        values_to_save={"epochs": max_epochs},
        model=model,
        optimizer=optimizer,
    )

    # 5. Return a dictionary with the results, or a single float value (loss)
    return {
        "loss": val_loss,
        "info_dict": {
            "train_accuracy": 0.92,
            "test_accuracy": 0.72,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results",
        max_evaluations_total=15,
        searcher="priorband",
    )
