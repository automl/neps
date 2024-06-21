# Initializing the Pipeline Space

In NePS, we need to define a `pipeline_space`.
This space can be structured through various approaches, including a Python dictionary, a YAML file, or ConfigSpace.
Each of these methods allows you to specify a set of parameter types, ranging from Float and Categorical to specialized architecture parameters.
Whether you choose a dictionary, YAML file, or ConfigSpace, your selected method serves as a container or framework
within which these parameters are defined and organized. This section not only guides you through the process of
setting up your `pipeline_space` using these methods but also provides detailed instructions and examples on how to
effectively incorporate various parameter types, ensuring that NePS can utilize them in the optimization process.


## Parameters
NePS currently features 4 primary hyperparameter types:

* [`CategoricalParameter`][neps.search_spaces.hyperparameters.categorical.CategoricalParameter]
* [`FloatParameter`][neps.search_spaces.hyperparameters.float.FloatParameter]
* [`IntegerParameter`][neps.search_spaces.hyperparameters.integer.IntegerParameter]
* [`ConstantParameter`][neps.search_spaces.hyperparameters.constant.ConstantParameter]

Using these types, you can define the parameters that NePS will optimize during the search process.
The most basic way to pass these parameters is through a Python dictionary, where each key-value
pair represents a parameter name and its respective type.
For example, the following Python dictionary defines a `pipeline_space` with four parameters
for optimizing a deep learning model:

```python
pipeline_space = {
    "learning_rate": neps.FloatParameter(0.00001, 0.1, log=True),
    "num_epochs": neps.IntegerParameter(3, 30, is_fidelity=True),
    "optimizer": neps.CategoricalParameter(["adam", "sgd", "rmsprop"]),
    "dropout_rate": neps.ConstantParameter(0.5),
}

neps.run(.., pipeline_space=pipeline_space)
```

??? example "Quick Parameter Reference"

    === "`CategoricalParameter`"

        ::: neps.search_spaces.hyperparameters.categorical.CategoricalParameter

    === "`FloatParameter`"

        ::: neps.search_spaces.hyperparameters.float.FloatParameter

    === "`IntegerParameter`"

        ::: neps.search_spaces.hyperparameters.integer.IntegerParameter

    === "`ConstantParameter`"

        ::: neps.search_spaces.hyperparameters.constant.ConstantParameter


## Using your knowledge, providing a Prior
When optimizing, you can provide your own knowledge using the parameters `default=`.
By indicating a `default=` we take this to be your user prior,
**your knowledge about where a good value for this parameter lies**.

You can also specify a `default_confidence=` to indicate how strongly you want NePS,
to focus on these, one of either `"low"`, `"medium"`, or `"high"`.

Currently the two major algorithms that exploit this in NePS are `PriorBand`
(prior-based `HyperBand`) and `PiBO`, a version of Bayesian Optimization which uses Priors.

```python
import neps

neps.run(
    ...,
    pipeline_space={
        "learning_rate": neps.FloatParameter(1e-4, 1e-1, log=True, default=1e-2, default_confidence="medium"),
        "num_epochs": neps.IntegerParameter(3, 30, is_fidelity=True),
        "optimizer": neps.CategoricalParameter(["adam", "sgd", "rmsprop"], default="adam", default_confidence="low"),
        "dropout_rate": neps.ConstantParameter(0.5),
    }
)
```
!!! warning "Must set `default=` for all parameters, if any"

    If you specify `default=` for one parameter, you must do so for all your variables.
    This will be improved in future versions.

!!! warning "Interaction with `is_fidelity`"

    If you specify `is_fidelity=True` for one parameter, the `default=` and `default_confidence=` are ignored.
    This will be dissallowed in future versions.

## Defining a pipeline space using YAML
Create a YAML file (e.g., `./pipeline_space.yaml`) with the parameter definitions following this structure.

=== "`./pipeline_space.yaml`"

    ```yaml
    learning_rate:
      type: float
      lower: 2e-3
      upper: 0.1
      log: true

    num_epochs:
      type: int
      lower: 3
      upper: 30
      is_fidelity: true

    optimizer:
      type: categorical
      choices: ["adam", "sgd", "rmsprop"]

    dropout_rate: 0.5
    ```

=== "`run.py`"

    ```python
    neps.run(.., pipeline_space="./pipeline_space.yaml")
    ```

When defining the `pipeline_space` using a YAML file, if the `type` argument is not specified,
the NePS will automatically infer the data type based on the value provided.

* If `lower` and `upper` are provided, then if they are both integers, the type will be inferred as `int`,
    otherwise as `float`. You can provide scientific notation for floating-point numbers as well.
* If `choices` are provided, the type will be inferred as `categorical`.
* If just a numeric or string is provided, the type will be inferred as `constant`.

If none of these hold, an error will be raised.


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

!!! warning

    Parameters you wish to use as a **fidelity** are not support through ConfigSpace
    at this time.

For additional information on ConfigSpace and its features, please visit the following
[link](https://github.com/automl/ConfigSpace).

## Supported Architecture parameter Types
A comprehensive documentation for the Architecture parameter is not available at this point.

If you are interested in exploring architecture parameters, you can find detailed
examples and usage in the following resources:

- [Basic Usage Examples](https://github.com/automl/neps/tree/master/neps_examples/basic_usage) - Basic usage
    examples that can help you understand the fundamentals of Architecture parameters.
- [Experimental Examples](https://github.com/automl/neps/tree/master/neps_examples/experimental) - For more advanced
    and experimental use cases, including Hierarchical parameters, check out this collection of examples.

!!! warning

    The configuration of `pipeline_space` from a YAML file does not currently support architecture parameter types.
