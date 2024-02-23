# Initializing the Pipeline Space

In NePS, a pivotal step is the definition of the search space, termed `pipeline_space`. This space can be structured
through various approaches, including a Python dictionary, a YAML file, or ConfigSpace. Each of these methods allows
you to specify a set of parameter types, ranging from Float and Categorical to specialized architecture parameters.
Whether you choose a dictionary, YAML file, or ConfigSpace, your selected method serves as a container or framework
within which these parameters are defined and organized. This section not only guides you through the process of
setting up your `pipeline_space` using these methods but also provides detailed instructions and examples on how to
effectively incorporate various parameter types, ensuring that NePS can utilize them in the optimization process.


## Methods for Defining the NePS Pipeline Space
### Option 1: Using a Python Dictionary

To define the `pipeline_space` using a Python dictionary, follow these steps:

Create a Python dictionary that specifies the parameters and their respective ranges. For example:

```python
pipeline_space = {
    "learning_rate": neps.FloatParameter(lower=0.00001, upper=0.1, log=True),
    "num_epochs": neps.IntegerParameter(lower=3, upper=30, is_fidelity=True),
    "optimizer": neps.CategoricalParameter(choices=["adam", "sgd", "rmsprop"]),
    "dropout_rate": neps.FloatParameter(value=0.5),
}
```

### Option 2: Using a YAML File

Create a YAML file (e.g., pipeline_space.yaml) with the parameter definitions following this structure.

```yaml
pipeline_space: # important to start with
  learning_rate:
    lower: 2e-3
    upper: 0.1
    log: true

  num_epochs:
    type: int # or "integer", optional if u want to manually set this
    lower: 3
    upper: 30
    is_fidelity: True

  optimizer:
    choices: ["adam", "sgd", "rmsprop"]

  dropout_rate:
    value: 0.5
...
```

Ensure your YAML file starts with `pipeline_space:`.
This is the root key under which all parameter configurations are defined.

!!! note "Note"
    The various types of parameters displayed in the Dictionary of Option 1 are here automatically determined by the
    data. If desired, you have the option to define them manually by providing the argument `type`. For more details,
    refer to the section on [Supported Hyperparameter Types](#supported-hyperparameter-types).


### Option 3: Using ConfigSpace

For users familiar with the ConfigSpace library, can also define the `pipeline_space` through
ConfigurationSpace().

```python
from configspace import ConfigurationSpace, UniformFloatHyperparameter

configspace = ConfigurationSpace()
configspace.add_hyperparameter(
    UniformFloatHyperparameter("learning_rate", 0.00001, 0.1, log=True)
)
```

For additional information on ConfigSpace and its features, please visit the following
[link](https://github.com/automl/ConfigSpace).
## Supported Hyperparameter Types

### Float/Integer Parameter

- **Expected Arguments:**
    - `lower`: The minimum value of the parameter.
    - `upper`: The maximum value of the parameter.
        - Accepted values: int or float depending on the specific parameter type one wishes to use.
- **Optional Arguments:**
    - `log`: Boolean that indicates if the parameter uses a logarithmic scale (default: False)
        - [Details on how YAML interpret Boolean Values](#important-note-on-yaml-data-type-interpretation)
    - `is_fidelity`: Boolean that marks the parameter as a fidelity parameter (default: False).
    - `default`: Sets a prior central value for the parameter (default: None).
      > Note: Currently, if you define a prior for one parameter, you must do so for all your variables.
    - `default_confidence`: Specifies the confidence level of the default value,
      indicating how strongly the prior
      should be considered (default: 'low').
        - Accepted values: 'low', 'medium', or 'high'.
    - `type`: Specifies the data type of the parameter.
        - Accepted values: 'int', 'integer', or 'float'.
        > Note: If type is not specified e notation gets converted to float

        !!! note "YAML Method Specific:"
            The type argument, used to specify the data type of parameters as 'int', 'integer', or 'float',
            is unique to defining the pipeline_space with a YAML file. This explicit specification of the parameter
            type is not required when using a Python dictionary or ConfigSpace, as these methods inherently determine
            the data types based on the syntax and structure of the code.

### Categorical Parameter

- **Expected Arguments:**
    - `choices`: A list of discrete options (int | float | str) that the parameter can take.
- **Optional Arguments:**
    - `is_fidelity`: Marks the parameter as a fidelity parameter (default: False).
        - [Details on how YAML interpret Boolean Values](#important-note-on-yaml-data-type-interpretation)
    - `default`: Sets a prior central value for the parameter (default: None).
      > Note: Currently, if you define a prior for one parameter, you must do so for all your variables.
    - `default_confidence`: Specifies the confidence level of the default value,
      indicating how strongly the prior
      should be considered (default: "low").
      - `type`: Specifies the data type of the parameter.
        - Accepted values: 'cat' or 'categorical'.
        > Note: Yaml Method Specific

### Constant Parameter

- **Expected Arguments:**
    - `value`: The fixed value (int | float | str) for the parameter.
- **Optional Arguments:**
    - `type`: Specifies the data type of the parameter.
        - Accepted values: 'const' or 'constant'.
      > Note: Yaml Method Specific
    - `is_fidelity`: Marks the parameter as a fidelity parameter (default: False).

### Important Note on YAML Data Type Interpretation

When working with YAML files, it's essential to understand how the format interprets different data types:

1. **Strings in Quotes:**

    - Any value enclosed in single (`'`) or double (`"`) quotes is treated as a string.
    - Example: `"true"`, `'123'` are read as strings.

2. **Boolean Interpretation:**

    -  Specific unquoted values are interpreted as booleans. This includes:
        - `true`, `True`, `TRUE`
        - `false`, `False`, `FALSE`
        - `on`, `On`, `ON`
        - `off`, `Off`, `OFF`
        - `yes`, `Yes`, `YES`
        - `no`, `No`, `NO`

3. **Numbers:**

    - Unquoted numeric values are interpreted as integers or floating-point numbers, depending on their format.
    - By default, when the 'type' is not specified, any number in scientific notation (e.g., 1e3) is interpreted as a
   floating-point number. This interpretation is unique to our system.

4. **Empty Strings:**

    - An empty string `""` or a key with no value is always treated as `null` in YAML.

5. **Unquoted Non-Boolean, Non-Numeric Strings:**

    - Unquoted values that don't match boolean patterns or numeric formats are treated as strings.
    - Example: `example` is a string.

Remember to use appropriate quotes and formats to ensure values are interpreted as intended.

## Supported Architecture parameter Types

!!! note "Note"
    The configuration of `pipeline_space` from a YAML file does not currently support architecture parameter types.
!!! note "Note"
    A comprehensive documentation for the Architecture parameter will be available soon.
    If you are interested in exploring architecture parameters, you can find detailed
    examples and usage in the following resources:

    - [Basic Usage Examples](https://github.com/automl/neps/tree/master/neps_examples/basic_usage) - Basic usage
        examples that can help you understand the fundamentals of Architecture parameters.
    - [Experimental Examples](https://github.com/automl/neps/tree/master/neps_examples/experimental) - For more advanced
        and experimental use cases, including Hierarchical parameters, check out this collection of examples.
