"""
NOTE!!! This code is not meant to be run directly and is only used for demonstration
purposes.

Steps for a successful neps run:
1. Importing neccessary packages including neps-
2. Designing the search space as a dicionray.
3. Creating the run_pipeline and returning the loss and other wanted metrics.
4. Using neps run with the optimizer of choice.
"""

import logging

import neps


def pipeline_space() -> dict:
    # Create the search space based on NEPS parameters and return the dictionary.
    # Example:
    space = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True),
    )
    return space


def run_pipeline(**config) -> dict | float:
    # Run pipeline should include the following steps:

    # 1. Defining the model.
    # 2. Each optimization variable should get its values from the pipeline space.
    #   Example:
    #   learning_rate = config["lr"]
    # 3. The training loop
    # 4. Returning the loss, which can be either as a single float or as part of
    #   an info dictionary containing other metrics.
    return dict or float


if __name__ == "__main__":
    # 1. Creating the logger
    logging.basicConfig(level=logging.INFO)
    # 2. Passing the correct arguments to the neps.run function
    # For more information on the searcher, please take a look at this link:
    # https://github.com/automl/neps/tree/master/src/neps/optimizers
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results",
        max_evaluations_total=10,
    )
