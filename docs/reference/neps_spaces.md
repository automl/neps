# NePS Spaces

**NePS Spaces** provide a powerful framework for defining and optimizing complex search spaces across the entire pipeline, including [hyperparameters](#1-constructing-hyperparameter-spaces), [architecture search](#3-architectures) and [more](#4-general-structures).

## 1. Constructing Hyperparameter Spaces

**NePS spaces** include all the necessary components to define a Hyperparameter Optimization (HPO) search space like:

- [`neps.Integer`][neps.space.neps_spaces.parameters.Integer]: Discrete integer values
- [`neps.Float`][neps.space.neps_spaces.parameters.Float]: Continuous float values
- [`neps.Categorical`][neps.space.neps_spaces.parameters.Categorical]: Discrete categorical values
- [`neps.Fidelity`][neps.space.neps_spaces.parameters.Fidelity]: Special type for float or integer, [multi-fidelity](../reference/search_algorithms/multifidelity.md) parameters (e.g., epochs, dataset size)

Using these types, you can define the parameters that NePS will optimize during the search process.
A **NePS space** is defined as a subclass of [`PipelineSpace`][neps.space.neps_spaces.parameters.PipelineSpace]. Here we define the hyperparameters that make up the space, like so:

```python
import neps

class MySpace(neps.PipelineSpace):
    float_param = neps.Float(min_value=0.1, max_value=1.0)
    int_param = neps.Integer(min_value=1, max_value=10)
    cat_param = neps.Categorical(choices=("A", "B", "C"))
```

!!! info "Using **NePS Spaces**"

    To search a **NePS space**, pass it as the `pipeline_space` argument to the `neps.run()` function:

    ```python
    neps.run(
        ...,
        pipeline_space=MySpace()
    )
    ```

    For more details on how to use the `neps.run()` function, see the [NePS Run Reference](../reference/neps_run.md).

### Using cheap approximation, providing a [**Fidelity**](../reference/search_algorithms/landing_page_algo.md#what-is-multi-fidelity-optimization) Parameter

Passing a [`neps.Integer`][neps.space.neps_spaces.parameters.Integer] or [`neps.Float`][neps.space.neps_spaces.parameters.Float] to a [`neps.Fidelity`][neps.space.neps_spaces.parameters.Fidelity] allows you to employ multi-fidelity optimization strategies, which can significantly speed up the optimization process by evaluating configurations at different fidelities (e.g., training for fewer epochs):

```python
epochs = neps.Fidelity(neps.Integer(1, 16))
```

For more details on how to use fidelity parameters, see the [Multi-Fidelity](../reference/search_algorithms/landing_page_algo.md#what-is-multi-fidelity-optimization) section.

### Using your knowledge, providing a [**Prior**](../reference/search_algorithms/landing_page_algo.md#what-are-priors)

You can provide **your knowledge about where a good value for this parameter lies** by indicating a `prior=`. You can also specify a `prior_confidence=` to indicate how strongly you want NePS to focus on these, one of either `"low"`, `"medium"`, or `"high"`:

```python
# Here "A" is used as a prior, indicated by its index 0
cat_with_prior = neps.Categorical(choices=("A", "B", "C"), prior=0, prior_confidence="high")
```

For more details on how to use priors, see the [Priors](../reference/search_algorithms/landing_page_algo.md#what-are-priors) section.

## 3. Constructing Architecture Spaces

Additionally, **NePS spaces** can describe **complex (hierarchical) architectures** using:

- [`Operation`][neps.space.neps_spaces.parameters.Operation]: Define operations and their arguments

Operations can be Callables, (e.g. pytorch objects) which will be passed to the evaluation function as such:

```python

import torch.nn

class NNSpace(PipelineSpace):

    # Defining operations for different activation functions
    _relu = Operation(operator=torch.nn.ReLU)
    _sigmoid = Operation(operator=torch.nn.Sigmoid)

    # We can then search over these operations and use them in the evaluation function
    activation_function = neps.Categorical(choices=(_relu, _sigmoid))
```

!!! info "Intermediate parameters"

    When defining parameters that should not be passed to the evaluation function and instead are used in other parameters, prefix them with an underscore, like here in `_layer_size`. Otherwise this might lead to `unexpected arguments` errors.

Operation also allow for (keyword-)arguments to be defined, including other parameters of the space:

```python

    _layer_size = neps.Integer(min_value=80, max_value=100)

    hidden_layer = neps.Operation(
        operator=torch.nn.Linear,
        kwargs={"input_size": 64,               # Fixed input size
                "output_size": _layer_size},    # Using the previously defined parameter

        # Or for non-keyword arguments:
        args=(activation_function,)
    )
```

This can be used for efficient architecture search by defining cells and blocks of operations, that make up a neural network.

??? abstract "Structural Space-compatible optimizers"

    Currently, NePS Spaces is compatible with these optimizers, which can be imported from [neps.algorithms][neps.optimizers.algorithms--neps-algorithms]:

    - [`Random Search`][neps.optimizers.algorithms.random_search], which can sample the space uniformly at random
    - [`Complex Random Search`][neps.optimizers.algorithms.complex_random_search], which can sample the space uniformly at random, using priors and mutating previously sampled configurations
    - [`PriorBand`][neps.optimizers.algorithms.priorband], which uses [multi-fidelity](./search_algorithms/multifidelity.md) and the prior knowledge encoded in the NePS space

## 4. Constructing Complex Spaces

Until now all parameters are sampled once and their value used for all occurrences. This section describes how to resample parameters in different contexts using:

- [`neps.Resampled`][neps.space.neps_spaces.parameters.Resampled]: Resample from an existing parameters range

With `neps.Resampled` you can reuse a parameter, even themselves recursively, but with a new value each time:

```python
    float_param = neps.Float(min_value=0, max_value=1)

    # The resampled parameter will have the same range but will be sampled
    # independently, so it can take a different value than its source
    resampled_float = neps.Resampled(source=float_param)
```

This is especially useful for defining complex architectures, where e.g. a cell block is defined and then resampled multiple times to create a neural network architecture:

```python

    _kernel_size = neps.Integer(min_value=5, max_value=8)

    # Define a cell block that can be resampled
    # It will resample a new kernel size from _kernel_size each time
    _cell_block = neps.Operation(
        operator=torch.nn.Conv2d,
        kwargs={"kernel_size": neps.Resampled(source=_kernel_size)}
    )

    # Resample the cell block multiple times to create a convolutional neural network
    cnn = torch.nn.Sequential(
        neps.Resampled(_cell_block),
        neps.Resampled(_cell_block),
        neps.Resampled(_cell_block),
    )
```

??? info "Self- and future references"

    When referencing itself or a not yet defined parameter (to enable recursions) use a string of that parameters name:

    ```python
    self_reference = Categorical(
        choices=(
            # It will either choose to resample itself twice
            (Resampled("self_reference"), Resampled("self_reference")),
            # Or it will sample the future parameter
            (Resampled("future_param"),),
        )
    )
    # This results in a (possibly infinite) tuple of independently sampled future_params

    future_param = Float(min_value=0, max_value=5)
    ```

!!! tip "Complex structural spaces"

    Together, [Resampling][neps.space.neps_spaces.parameters.Resampled] and [operations][neps.space.neps_spaces.parameters.Operation] allow you to define complex search spaces across the whole ML-pipeline akin to [Context-Free Grammars (CFGs)](https://en.wikipedia.org/wiki/Context-free_grammar), exceeding architecture search. For example, you can sample neural optimizers from a set of instructions, as done in [`NOSBench`](https://openreview.net/pdf?id=5Lm2ghxMlp) to train models.

## Inspecting Configurations

NePS saves the configurations as paths, where each sampling decision is recorded. As they are hard to read, so you can load the configuration from the `results/.../configs` directory using the [`NepsCompatConverter`][neps.space.neps_spaces.neps_space.NepsCompatConverter] class, which converts the configuration such that it can be used with the NePS Spaces API:

```python
from neps.space.neps_spaces import neps_space
import yaml

with open("Path/to/config.yaml", "r") as f:
    conf_dict = yaml.safe_load(f)
config = NepsCompatConverter.from_neps_config(conf_dict)

# Use the resolution context to sample the configuration using a
# Sampler that follows the instructions in the configuration
resolved_pipeline, resolution_context = neps_space.resolve(pipeline=NN_Space(),
    # Predefined samplings are the decisions made at each sampling step
    domain_sampler=neps_space.OnlyPredefinedValuesSampler(predefined_samplings=config.predefined_samplings),
    # Environment values are the fidelities and any arguments of the evaluation function not part of the search space
    environment_values=config.environment_values)

# The resolved_pipeline now contains all the parameters and their values, e.g. the Callable model
model_callable = neps_space.convert_operation_to_callable(operation=resolved_pipeline.model)
```

## Using ConfigSpace

For users familiar with the [`ConfigSpace`](https://automl.github.io/ConfigSpace/main/) library,
can also define the `pipeline_space` through `ConfigurationSpace()`

```python
from configspace import ConfigurationSpace, Float

configspace = ConfigurationSpace(
    {
        "learning_rate": Float("learning_rate", bounds=(1e-4, 1e-1), log=True)
        "optimizer": ["adam", "sgd", "rmsprop"],
        "dropout_rate": 0.5,
    }
)
```
