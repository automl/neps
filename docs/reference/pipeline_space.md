# Initializing the Pipeline Space

In NePS, we need to define a `pipeline_space`.
This space can be structured through various approaches, including a Python dictionary, or ConfigSpace.
Each of these methods allows you to specify a set of parameter types, ranging from Float and Categorical to specialized architecture parameters.
Whether you choose a dictionary, or ConfigSpace, your selected method serves as a container or framework
within which these parameters are defined and organized. This section not only guides you through the process of
setting up your `pipeline_space` using these methods but also provides detailed instructions and examples on how to
effectively incorporate various parameter types, ensuring that NePS can utilize them in the optimization process.


## Parameters
NePS currently features 4 primary hyperparameter types:

* [`Categorical`][neps.space.Categorical]
* [`Float`][neps.space.Float]
* [`Integer`][neps.space.Integer]
* [`Constant`][neps.space.Constant]

Using these types, you can define the parameters that NePS will optimize during the search process.
The most basic way to pass these parameters is through a Python dictionary, where each key-value
pair represents a parameter name and its respective type.
For example, the following Python dictionary defines a `pipeline_space` with four parameters
for optimizing a deep learning model:

```python
pipeline_space = {
    "learning_rate": neps.Float(0.00001, 0.1, log=True),
    "num_epochs": neps.Integer(3, 30, is_fidelity=True),
    "optimizer": ["adam", "sgd", "rmsprop"], # Categorical
    "dropout_rate": 0.5, # Constant
}

neps.run(.., pipeline_space=pipeline_space)
```

??? example "Quick Parameter Reference"

    === "`Categorical`"

        ::: neps.space.Categorical

    === "`Float`"

        ::: neps.space.Float

    === "`Integer`"

        ::: neps.space.Integer

    === "`Constant`"

        ::: neps.space.Constant


## Using your knowledge, providing a Prior
When optimizing, you can provide your own knowledge using the parameter `prior=`.
By indicating a `prior=` we take this to be your user prior,
**your knowledge about where a good value for this parameter lies**.

You can also specify a `prior_confidence=` to indicate how strongly you want NePS,
to focus on these, one of either `"low"`, `"medium"`, or `"high"`.

```python
import neps

neps.run(
    ...,
    pipeline_space={
        "learning_rate": neps.Float(1e-4, 1e-1, log=True, prior=1e-2, prior_confidence="medium"),
        "num_epochs": neps.Integer(3, 30, is_fidelity=True),
        "optimizer": neps.Categorical(["adam", "sgd", "rmsprop"], prior="adam", prior_confidence="low"),
        "dropout_rate": neps.Constant(0.5),
    }
)
```

!!! warning "Interaction with `is_fidelity`"

    If you specify `is_fidelity=True` and `prior=` for one parameter, this will raise an error.

Currently the two major algorithms that exploit this in NePS are `PriorBand`
(prior-based `HyperBand`) and `PiBO`, a version of Bayesian Optimization which uses Priors. For more information on priors and algorithms using them, please refer to the [prior documentation](../reference/search_algorithms/prior.md).

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
