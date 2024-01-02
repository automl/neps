# The run_pipeline Function

The `run_pipeline` function is crucial for NePS. It encapsulates the `objective function` to be minimized, which could be a regular equation or a neural network.

This function receives the configuration to be utilized from the parameters defined in the search space. Consequently, it executes the same set of instructions or equations based on the provided configuration to minimize the objective function.

The `run_pipeline` function will look similar to the following:

```python
def run_pipeline(
    pipeline_directory,           # The directory where the config is saved
    previous_pipeline_directory,  # The directory of the immediate lower fidelity config
    **config,                     # The hyperparameters to be used in the pipeline
):

    element_1 = config[element_1]
    element_2 = config[element_2]
    element_3 = config[element_3]

    loss = element_1 - element_2 + element_3

    return loss
```

The `run_pipeline` function should be replaced with the user's specific objective function. It is invoked by `neps.run` without any arguments, as these arguments are automatically handled by NePS. Additionally, NePS provides the pipeline directory and the previous pipeline directory for user convenience (mainly useful for searches that require fidelities).

Have a look at our examples and templates [here](https://github.com/automl/neps/tree/master/neps_examples) to see how we use this function in different scenarios.

