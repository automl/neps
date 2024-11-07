"""
NOTE!!! This code is not meant to be executed.
It is only to serve as a template to help interface NePS with an existing ML/DL pipeline.

The following script is designed as a template for using NePS.
It describes the crucial components that a user needs to provide in order to interface
a NePS optimizer.

The 2 crucial components are:
* The search space, called the `pipeline_space` in NePS
  * This defines the set of hyperparameters that the optimizer will search over
  * This declaration also allows injecting priors in the form of defaults per hyperparameter
* The `evaluate_pipeline` function
  * This function is called by the optimizer and is responsible for running the pipeline
  * The function should at the minimum expect the hyperparameters as keyword arguments
  * The function should return the objective_to_minimize of the pipeline as a float
    * If the return value is a dictionary, it should have a key called "objective_to_minimize" with the objective_to_minimize as a float


Overall, running an optimizer from NePS involves 4 clear steps:
1. Importing neccessary packages including neps.
2. Designing the search space as a dictionary.
3. Creating the evaluate_pipeline and returning the objective_to_minimize and other wanted metrics.
4. Using neps run with the optimizer of choice.
"""

import logging
from warnings import warn

import neps


logger = logging.getLogger("neps_template.run")


def pipeline_space() -> dict:
    # Create the search space based on NEPS parameters and return the dictionary.
    # Example:
    space = dict(
        lr=neps.Float(
            lower=1e-5,
            upper=1e-2,
            log=True,      # If True, the search space is sampled in log space
            default=1e-3,  # a non-None value here acts as the mode of the prior distribution
        ),
    )
    return space

def run_pipeline(**config) -> dict | float:
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(**config)

def evaluate_pipeline(**config) -> dict | float:
    # Run pipeline should include the following steps:

    # 1. Defining the model.
    # 1.1 Load any checkpoint if necessary
    # 2. Each optimization variable should get its values from the pipeline space.
    #   Example:
    #   learning_rate = config["lr"]
    # 3. The training loop
    # 3.1 Save any checkpoint if necessary
    # 4. Returning the objective_to_minimize, which can be either as a single float or as part of
    #   an info dictionary containing other metrics.

    # Can use global logger to log any information
    logger.info(f"Running pipeline with config: {config}")

    return dict or float


if __name__ == "__main__":
    # 1. Creating the logger


    # 2. Passing the correct arguments to the neps.run function
    # For more information on the searcher, please take a look at this link:
    # https://github.com/automl/neps/tree/master/neps/optimizers/README.md

    neps.run(
        evaluate_pipeline=evaluate_pipeline,        # User TODO (defined above)
        pipeline_space=pipeline_space(),  # User TODO (defined above)
        root_directory="results",
        max_evaluations_total=10,
    )
