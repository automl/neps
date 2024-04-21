# Configuring and Running Optimizations

The `neps.run` function is the core interface for running Hyperparameter and/or architecture search using optimizers in NePS.

This document breaks down the core arguments that allow users to control the optimization process in NePS. 

## Search Strategy
At default NePS intelligently selects the most appropriate search strategy based on your defined configurations in
`pipeline_space`.
The characteristics of your search space, as represented in the `pipeline_space`, play a crucial role in determining
which optimizer NePS will choose. This automatic selection process ensures that the strategy aligns perfectly
with the specific requirements and nuances of your search space, thereby optimizing the effectiveness of the
hyperparameter and/or architecture optimization. You can also manually select a specific or custom optimizer that better
matches your specific needs. For more information, refer [here](https://automl.github.io/neps/latest/optimizers).

## Arguments

::: neps.run


## Parallelization

`neps.run` can be called multiple times with multiple processes or machines, to parallelize the optimization process.
Ensure that `root_directory` points to a shared location across all instances to synchronize the optimization efforts.
For more information [look here](https://automl.github.io/neps/latest/parallelization)

## Customization

The `neps.run` function allows for extensive customization through its arguments, enabling to adapt the
optimization process to the complexities of your specific problems.

For a deeper understanding of how to use `neps.run` in a practical scenario, take a look at our
[examples and templates](https://github.com/automl/neps/tree/master/neps_examples).
