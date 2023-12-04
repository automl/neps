# Initializing the Search Space

In NePS, defining the Search Space is one of two essential tasks. You can define it either through a Python dictionary
,YAML file or ConfigSpace. This section provides examples and instructions for both methods.

## Option 1: Using a Python Dictionary

To define the Search Space using a Python dictionary, follow these steps:

Create a Python dictionary that specifies the parameters and their respective ranges. For example:

```python
search_space = {
    "learning_rate": neps.FloatParameter(lower=0.00001, upper=0.1, log=True),
    "num_epochs": neps.IntegerParameter(lower=3, upper=30, is_fidelity=True),
    "optimizer": neps.CategoricalParameter(choices=["adam", "sgd", "rmsprop"]),
    "dropout_rate": neps.FloatParameter(value=0.5),
}
```

## Option 2: Using a YAML File

Create a YAML file (e.g., search_space.yaml) with the parameter definitions following this structure.

```yaml
search_space: # important to start with
  learning_rate:
    lower: 2e-3 # or 2*10^-3
    upper: 0.1
    log: true

  num_epochs:
    lower: 3
    upper: 30
    is_fidelity: True

  optimizer:
    choices: ["adam", "sgd", "rmsprop"]

  dropout_rate:
    value: 0.5
...
```

Ensure your YAML file starts with `search_space:`.
This is the root key under which all parameter configurations are defined.

## Option 3: Using ConfigSpace

For users familiar with the ConfigSpace library, can also define the Search Space through
ConfigurationSpace()

```python
from configspace import ConfigurationSpace, UniformFloatHyperparameter

configspace = ConfigurationSpace()
configspace.add_hyperparameter(
    UniformFloatHyperparameter("learning_rate", 0.00001, 0.1, log=True)
)
```

For additional information on ConfigSpace and its features, please visit the following link:
https://github.com/automl/ConfigSpace

# Supported HyperParameter Types

### FloatParameter and IntegerParameter

- **Expected Arguments:**
  - `lower`: The minimum value of the parameter.
  - `upper`: The maximum value of the parameter.
- **Optional Arguments:**
  - `type`: Specifies the data type of the parameter. Accepted Values: 'int', 'integer', or 'float'.
  - `log`: Indicates if the parameter uses a logarithmic scale (default: False).
  - `is_fidelity`: Marks the parameter as a fidelity parameter (default: False).
  - `default`: Sets a prior central value for the parameter (default: None).
  - `default_confidence`: Specifies the confidence level of the default value,
    indicating how strongly the prior
    should be considered default: "low".

### Categorical Parameter

- **Expected Arguments:**
  - `choices`: A list of discrete options that the parameter can take.
- **Optional Arguments:**
  - `type`: Specifies the data type of the parameter. Accepted Values: 'cat' or 'categorical'.
  - `is_fidelity`: Marks the parameter as a fidelity parameter (default: False).
  - `default`: Sets a prior central value for the parameter (default: None.
  - `default_confidence`: Specifies the confidence level of the default value,
    indicating how strongly the prior
    should be considered default: "low".

### ConstantParameter

- **Expected Arguments:**
  - `value`: The fixed value for the parameter.
- **Optional Arguments:**
  - `type`: Specifies the data type of the parameter. Accepted Values: 'const' or 'constant'.
  - `is_fidelity`: Marks the parameter as a fidelity parameter (default: False).

# Supported ArchitectureParameter Types

**Note**: The definition of Search Space from a YAML file is limited to supporting only Hyperparameter Types.

If you are interested in exploring Architecture, particularly Hierarchical parameters, you can find detailed examples and usage in the following resources:

- [Basic Usage Examples](https://github.com/automl/neps/tree/master/neps_examples/basic_usage) - This link provides basic usage examples that can help you understand the fundamentals of Architecture parameters.

- [Experimental Examples](https://github.com/automl/neps/tree/master/neps_examples/experimental) - For more advanced and experimental use cases, including Hierarchical parameters, check out this collection of examples.
