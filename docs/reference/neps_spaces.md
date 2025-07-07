# NePS Spaces: Joint Architecture and Hyperparameter Search

NePS Spaces provides a powerful framework for defining and optimizing complex search spaces, enabling both and joint architecture and hyperparameter search (JAHS).

## Constructing NePS Spaces

**NePS spaces** include all the necessary components to define a [Hyperparameter Optimization (HPO) search space](#hpo-search-spaces) like:

- [`Integer`][neps.space.neps_spaces.parameters.Integer]: Discrete integer values
- [`Float`][neps.space.neps_spaces.parameters.Float]: Continuous float values
- [`Categorical`][neps.space.neps_spaces.parameters.Categorical]: Discrete categorical values
- [`Fidelity`][neps.space.neps_spaces.parameters.Fidelity]: Special type for float or integer, [multi-fidelity](../reference/search_algorithms/multifidelity.md) parameters (e.g., epochs, dataset size)

Additionally, **NePS spaces** can describe [complex (hierarchical) architectures](#hierarchies-and-architectures) using:

- [`Operation`][neps.space.neps_spaces.parameters.Operation]: Define operations (e.g., convolution, pooling, activation) with arguments
- [`Resampled`][neps.space.neps_spaces.parameters.Resampled]: Resample other parameters

### HPO Search Spaces

A **NePS space** is defined as a subclass of [`Pipeline`][neps.space.neps_spaces.parameters.Pipeline]:

```python

class pipeline_space(Pipeline):
```

Here we define the hyperparameters that make up the space, like so:

```python

    float_param = Float(min_value=0.1, max_value=1.0)
    int_param = Integer(min_value=1, max_value=10)
    cat_param = Categorical(choices=["A", "B", "C"])
```

!!! tip "**Using your knowledge, providing a Prior**"

    You can provide **your knowledge about where a good value for this parameter lies** by indicating a `prior=`. You can also specify a `prior_confidence=` to indicate how strongly you want NePS to focus on these, one of either `"low"`, `"medium"`, or `"high"`:

    ```python
        # Categorical parameters can also choose between other parameters
        # Here the float parameter (index 0) is used as a prior
        float_or_int = Categorical(choices=(float_param, int_param), prior=0, prior_confidence="high")
    ```

### Hierarchies and Architectures

[Resampling][neps.space.neps_spaces.parameters.Resampled] and [operations][neps.space.neps_spaces.parameters.Operation] allow you to define complex architectures akin to [Context-Free Grammars (CFGs)](https://en.wikipedia.org/wiki/Context-free_grammar).

With `Resampled` you can reuse parameters in for other parameters, even themselves recursively:

```python
    # The new parameter will have the same range but will be resampled
    # independently, so it can take different values than its source
    resampled_float = Resampled(source=float_param)

    # If you only use a parameter to resample from it later, prefix it with an underscore
    # This way, your evaluation function will not receive it as an argument
    _float = Float(min_value=1, max_value=3)
    resampled_float_2 = Resampled(source=_float)
```

??? info "Self- and future references"

    When referencing itself or a not yet defined parameter use a string of that parameters name:

    ```python
    self_reference = Categorical(choices=(Resampled("self_reference"), Resampled("next_param")))
    next_param = Float(min_value=0, max_value=5)
    ```

Operations can be Callables, (e.g. pytorch objects) whose arguments can themselves be parameters:

```python

from torch.nn import Sequential, Conv2d, ReLU

class NN_Space(Pipeline):

    # Define an operation for a ReLU activation
    _relu = Operation(operator=ReLU)

    # Define a convolution operation with an optimizable kernel size parameter
    _convolution = Operation(
        operator=Conv2d,
        kwargs={"kernel_size": Integer(min_value=1, max_value=10)}
        # You could also define _kernel_size separately and use Resampled
    )

    _model_args = Categorical(
        choices=(
            # The Sequential will either get a convolution followed by a ReLU
            (Resampled(_convolution), _relu,),
            # Or two (different, hence the resampling) convolutions
            (Resampled(_convolution), Resampled(_convolution)),
            # Or just a ReLU activation
            (_relu,),
        )
    )

    # Define a sequential operation, using the previously defined _model_args
    # This model will be the only parameter passed to the evaluation function
    model = Operation(
        operator=Sequential,
        args=_model_args
    )
```

??? warning "Tuples as choice"

    When using a tuple as one of the choices in a `Categorical`, all choices must be tuples, as in the example above with ```(_relu,)```.

## Using NePS Spaces

To use a NePS space, pass it as the `pipeline_space` argument to the `neps.run()` function:

```python
import neps
neps.run(
    ...,
    pipeline_space=NN_Space()
)
```

!!! abstract "NePS Space-compatible optimizers"

    Currently, NePS Spaces is compatible with these optimizers, which can be imported from [neps.optimizers.neps_algorithms][neps.optimizers.neps_algorithms--neps-algorithms]:

    - [`Random Search`][neps.optimizers.neps_algorithms.neps_random_search], which can sample the space uniformly at random
    - [`Complex Random Search`][neps.optimizers.neps_algorithms.neps_complex_random_search], which can sample the space uniformly at random, using priors and mutating previously sampled configurations
    - [`PriorBand`][neps.optimizers.neps_algorithms.neps_priorband], which uses [multi-fidelity](./search_algorithms/multifidelity.md) and the prior knowledge encoded in the NePS space

## Inspecting Configurations

NePS saves the configurations as paths, where each sampling decision is recorded. As they are hard to read, so you can load the configuration from the `results` directory using the [`NepsCompatConverter`][neps.space.neps_spaces.neps_space.NepsCompatConverter] class, which converts the configuration such that it can be used with the NePS Spaces API:

```python
from neps.space.neps_spaces import neps_space
import yaml

with open("Path/to/config.yaml", "r") as f:
    conf_dict = yaml.safe_load(f)
resolution_context = NepsCompatConverter.from_neps_config(conf_dict)

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
