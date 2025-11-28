# NePS Spaces

**NePS Spaces** provide a powerful framework for defining and optimizing complex search spaces across the entire pipeline, including [hyperparameters](#1-constructing-hyperparameter-spaces), [architecture search](#3-constructing-architecture-spaces) and [more](#4-constructing-complex-spaces).

## 1. Constructing Hyperparameter Spaces

**NePS spaces** include all the necessary components to define a Hyperparameter Optimization (HPO) search space like:

- [`neps.Integer`][neps.space.neps_spaces.parameters.Integer]: Discrete integer values
- [`neps.Float`][neps.space.neps_spaces.parameters.Float]: Continuous float values
- [`neps.Categorical`][neps.space.neps_spaces.parameters.Categorical]: Discrete categorical values
- [`neps.IntegerFidelity`][neps.space.neps_spaces.parameters.IntegerFidelity]: Integer [multi-fidelity](../reference/search_algorithms/multifidelity.md) parameters (e.g., epochs, batch size)
- [`neps.FloatFidelity`][neps.space.neps_spaces.parameters.FloatFidelity]: Float [multi-fidelity](../reference/search_algorithms/multifidelity.md) parameters (e.g., dataset subset ratio)
- [`neps.Fidelity`][neps.space.neps_spaces.parameters.Fidelity]: Generic fidelity type (use IntegerFidelity or FloatFidelity instead)

Using these types, you can define the parameters that NePS will optimize during the search process.
A **NePS space** is defined as a subclass of [`PipelineSpace`][neps.space.neps_spaces.parameters.PipelineSpace]. Here we define the hyperparameters that make up the space, like so:

```python
import neps

class MySpace(neps.PipelineSpace):
    float_param = neps.Float(lower=0.1, upper=1.0)
    int_param = neps.Integer(lower=1, upper=10)
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

You can use [`neps.IntegerFidelity`][neps.space.neps_spaces.parameters.IntegerFidelity] or [`neps.FloatFidelity`][neps.space.neps_spaces.parameters.FloatFidelity] to employ multi-fidelity optimization strategies, which can significantly speed up the optimization process by evaluating configurations at different fidelities (e.g., training for fewer epochs):

```python
# Convenient syntax (recommended)
epochs = neps.IntegerFidelity(lower=1, upper=16)
subset_ratio = neps.FloatFidelity(lower=0.1, upper=1.0)

# Alternative syntax (also works)
epochs = neps.IntegerFidelity(1, 16)
```

For more details on how to use fidelity parameters, see the [Multi-Fidelity](../reference/search_algorithms/landing_page_algo.md#what-is-multi-fidelity-optimization) section.

### Using your knowledge, providing a [**Prior**](../reference/search_algorithms/landing_page_algo.md#what-are-priors)

You can provide **your knowledge about where a good value for this parameter lies** by indicating a `prior=`. You can also specify a `prior_confidence=` to indicate how strongly you want NePS to focus on these, one of either `"low"`, `"medium"`, or `"high"`:

```python
# Here "A" is used as a prior, indicated by its index 0
cat_with_prior = neps.Categorical(choices=("A", "B", "C"), prior=0, prior_confidence="high")
```

For more details on how to use priors, see the [Priors](../reference/search_algorithms/landing_page_algo.md#what-are-priors) section.

!!! info "Adding and removing parameters from **NePS Spaces**"

    To add or remove parameters from a `PipelineSpace` after its definition, you can use the `add()` and `remove()` methods. Mind you, these methods do NOT modify the existing space in-place, but return a new instance with the modifications:

    ```python
    space = MySpace()
    # Adding a new parameter using add()
    larger_space = space.add(neps.Integer(lower=5, upper=15), name="new_int_param")
    # Removing a parameter by its name
    smaller_space = space.remove("cat_param")
    ```

## 3. Constructing Architecture Spaces

Additionally, **NePS spaces** can describe **complex (hierarchical) architectures** using:

- [`Operation`][neps.space.neps_spaces.parameters.Operation]: Define operations and their arguments

Operations can be Callables, (e.g. pytorch objects) which will be passed to the evaluation function as such:

```python

import torch.nn

class NNSpace(PipelineSpace):

    # Defining operations for different activation functions
    _relu = neps.Operation(operator=torch.nn.ReLU)
    _sigmoid = neps.Operation(operator=torch.nn.Sigmoid)

    # We can then search over these operations and use them in the evaluation function
    activation_function = neps.Categorical(choices=(_relu, _sigmoid))
```

!!! info "Intermediate parameters"

    When defining parameters that should not be passed to the evaluation function and instead are used in other parameters, prefix them with an underscore, like here in `_layer_size`. Otherwise this might lead to `unexpected arguments` errors.

Operation also allow for (keyword-)arguments to be defined, including other parameters of the space:

```python

    batch_size = neps.Categorical(choices=(16, 32, 64))

    _layer_size = neps.Integer(lower=80, upper=100)

    hidden_layer = neps.Operation(
        operator=torch.nn.Linear,
        kwargs={"input_size": 64,               # Fixed input size
                "output_size": _layer_size},    # Using the previously defined parameter

        # Or for non-keyword arguments:
        args=(activation_function,)
    )
```

This can be used for efficient architecture search by defining cells and blocks of operations, that make up a neural network.
The `evaluate_pipeline` function will receive the sampled operations as Callables, which can be used to instantiate the model:

```python
def evaluate_pipeline(
    activation_function: torch.nn.Module,
    batch_size: int,
    hidden_layer: torch.nn.Linear):

    # Instantiate the model using the sampled operations
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        hidden_layer,
        activation_function,
        torch.nn.Linear(in_features=hidden_layer.out_features, out_features=10)
    )

    # Use the model for training and return the validation accuracy
    model.train(batch_size=batch_size, ...)
    return model.evaluate(...).accuracy

```

??? abstract "Structural Space-compatible optimizers"

    Currently, NePS Spaces is compatible with these optimizers, which can be imported from [neps.algorithms][neps.optimizers.algorithms--neps-algorithms]:

    - [`Random Search`][neps.optimizers.algorithms.random_search], which can sample the space uniformly at random
    - [`Complex Random Search`][neps.optimizers.algorithms.complex_random_search], which can sample the space uniformly at random, using priors and mutating previously sampled configurations
    - [`PriorBand`][neps.optimizers.algorithms.priorband], which uses [multi-fidelity](./search_algorithms/multifidelity.md) and the prior knowledge encoded in the NePS space

## 4. Constructing Complex Spaces

Until now all parameters are sampled once and their value used for all occurrences. This section describes how to resample parameters in different contexts using:

- [`.resample()`][neps.space.neps_spaces.parameters.Resample]: Resample from an existing parameters range

With `.resample()` you can reuse a parameter, even themselves recursively, but with a new value each time:

```python
class ResampleSpace(neps.PipelineSpace):
    float_param = neps.Float(lower=0, upper=1)

    # The resampled parameter will have the same range but will be sampled
    # independently, so it can take a different value than its source
    resampled_float = float_param.resample()
```

This is especially useful for defining complex architectures, where e.g. a cell block is defined and then resampled multiple times to create a neural network architecture:

```python
class CNN_Space(neps.PipelineSpace):
    _kernel_size = neps.Integer(lower=5, upper=8)

    # Define a cell block that can be resampled
    # It will resample a new kernel size from _kernel_size each time
    # Each instance will be identically but independently sampled
    _cell_block = neps.Operation(
        operator=torch.nn.Conv2d,
        kwargs={"kernel_size": _kernel_size.resample()}
    )

    # Resample the cell block multiple times to create a convolutional neural network
    cnn = torch.nn.Sequential(
        _cell_block.resample(),
        _cell_block.resample(),
        _cell_block.resample(),
    )

def evaluate_pipeline(cnn: torch.nn.Module):
    # Use the cnn model for training and return the validation accuracy
    cnn.train(...)
    return cnn.evaluate(...).accuracy
```

??? info "Self- and future references"

    When referencing itself or a not yet defined parameter (to enable recursions) use a string of that parameters name with `neps.Resample("parameter_name")`, like so:

    ```python
    self_reference = Categorical(
        choices=(
            # It will either choose to resample itself twice
            (neps.Resample("self_reference"), neps.Resample("self_reference")),
            # Or it will sample the future parameter
            (neps.Resample("future_param"),),
        )
    )
    # This results in a (possibly infinite) tuple of independently sampled future_params

    future_param = Float(lower=0, upper=5)
    ```

!!! tip "Complex structural spaces"

    Together, [Resampling][neps.space.neps_spaces.parameters.Resample] and [operations][neps.space.neps_spaces.parameters.Operation] allow you to define complex search spaces across the whole ML-pipeline akin to [Context-Free Grammars (CFGs)](https://en.wikipedia.org/wiki/Context-free_grammar), exceeding architecture search. For example, you can sample neural optimizers from a set of instructions, as done in [`NOSBench`](https://openreview.net/pdf?id=5Lm2ghxMlp) to train models.

## Inspecting Configurations

NePS saves the configurations as paths, where each sampling decision is recorded. As they are hard to read, so you can load the configuration using `neps.load_config()`, which returns a dictionary with the resolved parameters and their values:

```python
import neps

pipeline = neps.load_config("Path/to/config.yaml", pipeline_space=SimpleSpace()) # or
pipeline = neps.load_config("Path/to/neps_folder", config_id="config_0", pipeline_space=SimpleSpace())

# The pipeline now contains all the parameters and their values the same way they would be given to the evaluate_pipeline, e.g. the callable model:
model = pipeline["model"]
```

### Loading the Search Space from Disk

NePS automatically saves the search space when you run an optimization. You can retrieve it later using `neps.load_pipeline_space()`:

```python
import neps

# Load the search space from a previous run
pipeline_space = neps.load_pipeline_space("Path/to/neps_folder")

# Now you can use it to inspect configurations, continue runs, or analysis
```

!!! note "Auto-loading"

    In most cases, you don't need to call `load_pipeline_space()` explicitly. When continuing a run, `neps.run()` automatically loads the search space from disk. See [Continuing Runs](neps_run.md#continuing-runs) for more details.

!!! tip "Reconstructing a Run"

    You can load both the search space and optimizer information to fully reconstruct a previous run. See [Reconstructing and Reproducing Runs](neps_run.md#reconstructing-and-reproducing-runs) for a complete example.

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
