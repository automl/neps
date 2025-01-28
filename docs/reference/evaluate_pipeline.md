# The run function

## Introduction

The `evaluate_pipeline=` function is crucial for NePS. It encapsulates the objective function to be minimized, which could range from a regular equation to a full training and evaluation pipeline for a neural network.

This function receives the configuration to be utilized from the parameters defined in the search space. Consequently, it executes the same set of instructions or equations based on the provided configuration to minimize the objective function.

We will show some basic usages and some functionalites this function would require for successful implementation.

## Types of Returns

### 1. Single Value

Assuming the `pipeline_space=` was already created (have a look at [pipeline space](./pipeline_space.md) for more details).
A `evaluate_pipeline=` function with an objective of minimizing the loss will resemble the following:

```python
def evaluate_pipeline(
    **config,   # The hyperparameters to be used in the pipeline
):
    element_1 = config["element_1"]
    element_2 = config["element_2"]
    element_3 = config["element_3"]

    loss = element_1 - element_2 + element_3

    return loss
```

### 2. Dictionary

In this section, we will outline the special variables that are expected to be returned when the `evaluate_pipeline=` function returns a dictionary.

#### Loss

One crucial return variable is the `loss`. This metric serves as a fundamental indicator for the optimizer. One option is to return a dictionary with the `loss` as a key, along with other user-chosen metrics.

!!! note

    Loss can be any value that is to be minimized by the objective function.

```python
def evaluate_pipeline(
    **config,   # The hyperparameters to be used in the pipeline
):

    element_1 = config["element_1"]
    element_2 = config["element_2"]
    element_3 = config["element_3"]

    loss = element_1 - element_2 + element_3
    reverse_loss = -loss

    return {
        "objective_to_minimize": loss,
        "info_dict": {
            "reverse_loss": reverse_loss
            ...
        }
    }
```

#### Cost

Along with the return of the `loss`, the `evaluate_pipeline=` function would optionally need to return a `cost` in certain cases. Specifically when the `max_cost_total` parameter is being utilized in the `neps.run` function.


!!! note

    `max_cost_total` sums the cost from all returned configuration results and checks whether the maximum allowed cost has been reached (if so, the search will come to an end).

```python
import neps
import logging


def evaluate_pipeline(
    **config,   # The hyperparameters to be used in the pipeline
):

    element_1 = config["element_1"]
    element_2 = config["element_2"]
    element_3 = config["element_3"]

    loss = element_1 - element_2 + element_3
    cost = 2

    return {
        "objective_to_minimize": loss,
        "cost": cost,
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space, # Assuming the pipeline space is defined
        root_directory="results/bo",
        max_cost_total=10,
        optimizer="bayesian_optimization",
    )
```

Each evaluation carries a cost of 2. Hence in this example, the Bayesian optimization search is set to perform 5 evaluations.

## Arguments for Convenience

NePS also provides the `pipeline_directory` and the `previous_pipeline_directory` as arguments in the `evaluate_pipeline=` function for user convenience.

Regard an example to be run with a multi-fidelity optimizer, some checkpointing would be advantageos such that one does not have to train the configuration from scratch when the configuration qualifies to higher fidelity brackets.

```python
def evaluate_pipeline(
    pipeline_directory,           # The directory where the config is saved
    previous_pipeline_directory,  # The directory of the immediate lower fidelity config
    **config,                     # The hyperparameters to be used in the pipeline
):
    # Assume element3 is our fidelity element
    element_1 = config["element_1"]
    element_2 = config["element_2"]
    element_3 = config["element_3"]

    # Load any saved checkpoints
    checkpoint_name = "checkpoint.pth"
    start_element_3 = 0

    if previous_pipeline_directory is not None:
        # Read in state of the model after the previous fidelity rung
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        prev_element_3 = checkpoint["element_3"]
    else:
        prev_element_3 = 0

    start_element_3 += prev_element_3

    loss = 0
    for i in range(start_element_3, element_3):
        loss += element_1 - element_2

    torch.save(
        {
            "element_3": element_3,
        },
        pipeline_directory / checkpoint_name,
    )

    return loss
```

This could allow the proper navigation to the trained models and further train them on higher fidelities without repeating the entire training process.
