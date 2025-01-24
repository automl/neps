"""Boilerplate code to optimize a simple PyTorch model using PriorBand.

NOTE!!! This code is not meant to be executed.
It is only to serve as a template to help interface NePS with an existing ML/DL pipeline.


The following script is designed as a template for using `PriorBand` from NePS.
It describes the crucial components that a user needs to provide in order to interface PriorBand.

The 2 crucial components are:
* The search space, called the `pipeline_space` in NePS
  * This defines the set of hyperparameters that the optimizer will search over
  * This declaration also allows injecting priors in the form of defaults per hyperparameter
* The `evaluate_pipeline` function
  * This function is called by the optimizer and is responsible for running the pipeline
  * The function should at the minimum expect the hyperparameters as keyword arguments
  * The function should return the loss of the pipeline as a float
    * If the return value is a dictionary, it should have a key called "objective_to_minimize" with the loss as a float


Overall, running an optimizer from NePS involves 4 clear steps:
1. Importing neccessary packages including neps.
2. Designing the search space as a dictionary.
3. Creating the evaluate_pipeline and returning the loss and other wanted metrics.
4. Using neps run with the optimizer of choice.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import neps
from neps.utils.common import load_checkpoint, save_checkpoint

logger = logging.getLogger("neps_template.run")


def pipeline_space() -> dict:
    # Create the search space based on NEPS parameters and return the dictionary.
    # IMPORTANT:
    space = dict(
        lr=neps.Float(
            lower=1e-5,
            upper=1e-2,
            log=True,  # If True, the search space is sampled in log space
            prior=1e-3,  # a non-None value here acts as the mode of the prior distribution
        ),
        wd=neps.Float(
            lower=0,
            upper=1e-1,
            log=True,
            prior=1e-3,
        ),
        epoch=neps.Integer(
            lower=1,
            upper=10,
            is_fidelity=True,  # IMPORTANT to set this to True for the fidelity parameter
        ),
    )
    return space


def evaluate_pipeline(
    pipeline_directory: Path,  # The directory where the config is saved
    previous_pipeline_directory: Path
    | None,  # The directory of the config's immediate lower fidelity
    **config,  # The hyperparameters to be used in the pipeline
) -> dict | float:
    # Defining the model
    #  Can define outside the function or import from a file, package, etc.
    class my_model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features=224, out_features=512)
            self.linear2 = nn.Linear(in_features=512, out_features=10)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    # Instantiates the model
    model = my_model()

    # IMPORTANT: Extracting hyperparameters from passed config
    learning_rate = config["lr"]
    weight_decay = config["wd"]

    # Initializing the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    ## Checkpointing
    # loading the checkpoint if it exists
    previous_state = load_checkpoint(  # predefined function from neps
        directory=previous_pipeline_directory,
        model=model,  # relies on pass-by-reference
        optimizer=optimizer,  # relies on pass-by-reference
    )
    # adjusting run budget based on checkpoint
    if previous_state is not None:
        epoch_already_trained = previous_state["epochs"]
        # + Anything else saved in the checkpoint.
    else:
        epoch_already_trained = 0
        # + Anything else with default value.

    # Extracting target epochs from config
    max_epochs = config["epoch"]

    # User TODO:
    #  Load relevant data for training and validation

    # Actual model training
    for epoch in range(epoch_already_trained, max_epochs):
        # Training loop
        ...
        # Validation loop
        ...
        logger.info(f"Epoch: {epoch}, Loss: {...}, Val. acc.: {...}")

    # Save the checkpoint data in the current directory
    save_checkpoint(
        directory=pipeline_directory,
        values_to_save={"epochs": max_epochs},
        model=model,
        optimizer=optimizer,
    )

    # Return a dictionary with the results, or a single float value (loss)
    return {
        "objective_to_minimize": ...,
        "info_dict": {
            "train_accuracy": ...,
            "test_accuracy": ...,
        },
    }


# end of evaluate_pipeline


if __name__ == "__main__":
    neps.run(
        evaluate_pipeline=evaluate_pipeline,  # User TODO (defined above)
        pipeline_space=pipeline_space(),  # User TODO (defined above)
        root_directory="results",
        max_evaluations_total=25,  # total number of times `evaluate_pipeline` is called
        optimizer="priorband",  # "priorband_bo" for longer budgets, and set `initial_design_size``
    )
