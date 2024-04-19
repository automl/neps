# Declarative Usage

The `neps.run` function is the core of the NePS optimization process, where the search for the best hyperparameters
and architectures takes place. This document outlines the arguments and options available within this function,
providing a detailed guide to customize the optimization process to your specific needs.

## Search Strategy
At default NePS intelligently selects the most appropriate search strategy based on your defined configurations in
`pipeline_space`.
The characteristics of your search space, as represented in the `pipeline_space`, play a crucial role in determining
which optimizer NePS will choose. This automatic selection process ensures that the strategy aligns perfectly
with the specific requirements and nuances of your search space, thereby optimizing the effectiveness of the
hyperparameter and/or architecture optimization. You can also manually select a specific or custom optimizer that better
matches your specific needs. For more information, refer [here](https://automl.github.io/neps/latest/optimizers).

## Arguments

### Mandatory Arguments
- **`run_pipeline`** (function): The objective function, targeted by NePS for minimization, by evaluation various
  configurations. It requires these configurations as input and should return either a dictionary or a sole loss
  value as the
output. For correct setup instructions, refer to [here](https://automl.github.io/neps/latest/run_pipeline)
- **`pipeline_space`** (dict | yaml | configspace): This defines the search space for the configurations from which the
  optimizer samples. It accepts either a dictionary with the configuration names as keys, a path to a YAML
  configuration file, or a configSpace.ConfigurationSpace object. For comprehensive information and examples,
  please refer to the detailed guide available [here](https://automl.github.io/neps/latest/pipeline_space)

- **`root_directory`** (str): The directory path where the information about the optimization and its progress gets
  stored. This is also used to synchronize multiple calls to run(.) for parallelization.

- **Budget**:
To define a budget, provide either or both of the following parameters:

    - **`max_evaluations_total`** (int, default: None): Specifies the total number of evaluations to conduct before
      halting the optimization process.
    - **`max_cost_total`** (int, default: None): Prevents the initiation of new evaluations once this cost
      threshold is surpassed. This requires adding a cost value to the output of the `run_pipeline` function,
      for example, return {'loss': loss, 'cost': cost}. For more details, please refer
      [here](https://automl.github/io/neps/latest/run_pipeline)

You have the option to configure all arguments using a YAML file. This can be done by specifying the file path with
the following argument:

- **`run_args`** (str): Path to the YAML file for argument configuration. See [setup details](#setup-run_args) for more.

!!! note "Note"
        Currently we have a strict usage for `run_args`. So you can define either all arguments by providing them
        directly to neps.run or via the yaml file. This might change in the future. If you use yaml, directly provided
        arguments get overwritten either by the defined yaml config or the default value.

### Optional Arguments
##### Further Monitoring Options
  - **`overwrite_working_directory`** (bool, default: False): When set to True, the working directory
    specified by
    `root_directory` will be
    cleared at the beginning of the run. This is e.g. useful when debugging a `run_pipeline` function.
  - **`post_run_summary`** (bool, default: False): When enabled, this option generates a summary CSV file
    upon the
    completion of the
    optimization process. The summary includes details of the optimization procedure, such as the best configuration,
    the number of errors occurred, and the final performance metrics.
  - **`development_stage_id`** (int | float | str, default: None): An optional identifier used when working with
    multiple development stages. Instead of creating new root directories, use this identifier to save the results
    of an optimization run in a separate dev_id folder within the root_directory.
  - **`task_id`** (int | float | str, default: None): An optional identifier used when the optimization process
    involves multiple tasks. This functions similarly to `development_stage_id`, but it creates a folder named
    after the task_id instead of dev_id, providing an organized way to separate results for different tasks within
    the `root_directory`.
##### Parallelization Setup
  - **`max_evaluations_per_run`** (int, default: None): Limits the number of evaluations for this specific call of
    `neps.run`.
  - **`continue_until_max_evaluation_completed`** (bool, default: False): In parallel setups, pending evaluations
    normally count towards max_evaluations_total, halting new ones when this limit is reached. Setting this to
    True enables continuous sampling of new evaluations until the total of completed ones meets max_evaluations_total,
    optimizing resource use in time-sensitive scenarios.

For an overview and further resources on how NePS supports parallelization in distributed systems, refer to
the [Parallelization Overview](#parallelization).
##### Handling Errors
  - **`loss_value_on_error`** (float, default: None): When set, any error encountered in an evaluated configuration
    will not halt the process; instead, the specified loss value will be used for that configuration.
  - **`cost_value_on_error`** (float, default: None): Similar to `loss_value_on_error`, but for the cost value.
  - **`ignore_errors`** (bool, default: False): If True, errors encountered during the evaluation of configurations
    will be ignored, and the optimization will continue. Note: This error configs still count towards
    max_evaluations_total.
##### Search Strategy Customization
  - **`searcher`** (Literal["bayesian_optimization", "hyperband",..] | BaseOptimizer, default: "default"): Specifies
    manually which of the optimization strategy to use. Provide a string identifying one of the built-in
    search strategies or an instance of a custom `BaseOptimizer`.
  - **`searcher_path`** (Path | str, default: None): A path to a custom searcher implementation.
  - **`searcher_kwargs`**: Additional keyword arguments to be passed to the searcher.


  For more information about the available searchers and how to customize your own, refer
[here](https://automl.github.io/neps/latest/optimizers).
##### Others
  - **`pre_load_hooks`** (Iterable, default: None): A list of hook functions to be called before loading results.

## Setup `run_args`

To configure the execution parameters, you can also define them via a YAML file provided through the `run_args`
argument. Parameters not explicitly defined within this file will receive their default values. Below is
an example of how to structure your YAML configuration.

```yaml
run_args:  # Essential starting key; everything under this key gets considered for configuration.
  run_pipeline:
    path: "path/to/your/run_pipeline.py" # File path of the run_pipeline function
    name: "name_of_your_run_pipeline" # Function name
  pipeline_space: "path/to/your/search_space.yaml" # Path of the search space yaml file
  root_directory: "neps_results"  # Output directory for results
  max_evaluations_total: 100
  post_run_summary: # Defaults applied if left empty
  searcher: "bayesian_optimization"
  searcher_kwargs:
    initial_design_size: 5
    surrogate_model: "gp"
...
```
Once you've configured your YAML file as shown above, execute your setup by passing the file path to neps.run like this:
```python
neps.run(run_args="path/to/your/config.yaml")
```
### Special Loading Procedures
For defining `run_args`, some settings like `run_pipeline`, `pre_load_hooks`, and custom `searcher` need both a path
and a name to dynamically load them. This is crucial for components where NePS must find and execute specific code.
Below, the configurations for dynamic loading are outlined:

- **`run_pipeline`**:
    Specify the path and name of the pipeline function. This information allows NePS to find and initiate the desired
    pipeline.
    ```yaml
    run_pipeline:
      path: <path_to_run_pipeline>
      name: <run_pipeline_name>
    ```


- **`pipeline_space`**:
The pipeline_space configuration supports two modes of definition: directly in a YAML file or through a Python
dictionary that NePS can dynamically load.
      - **Direct YAML File Path**: Simply specify the path to your search space YAML.
        ```yaml
        pipeline_space: "path/to/your/search_space.yaml"
        ```
      - **Python Dictionary**: To define pipeline_space as a Python dictionary, specify the file path containing the
        dictionary and its name. This enables dynamic loading by NePS.
          ```yaml
          pipeline_space:
            path: <path_to_pipeline_space>
            name: <pipeline_space_name>
          ```


- **`searcher`**:
Configure the searcher to either utilize a built-in searcher or integrate a custom searcher of your creation.
      - **Built-in Searcher**: For using an already integrated searcher, simply specify its key. This is the
        straightforward method for standard optimization scenarios.
          ```yaml
          searcher: "bayesian_optimization" # key of the searcher
          ```
      - **Custom Searcher**: If you've developed a custom optimization algorithm and wish to employ it within NePS,
        specify the path and name of your custom searcher class. This allows NePS to dynamically load and use your
        custom implementation.
          ```yaml
          searcher:
            path: <path_to_custom_searcher_class>
            name: <custom_searcher_class_name>
          ```


- **`pre_load_hooks`**:
    Define hooks to be loaded before the main execution process begins. Assign each hook a unique identifier (e.g.
    hook1, hook2..) and detail the path to its implementation file along with the function name.
    This setup enables NePS to dynamically import and run these hooks during initialization.
        ```yaml
        pre_load_hooks:
          hook1:  # # Assign any unique key.
            path: <path_to_hook>  # Path to the hook's implementation file.
            name: <function_name>  # Name of the hook.
          hook2:
            path: <path_to_hook>
            name: <function_name>
        ```


For detailed examples and specific use case configurations,
visit [here](https://github.com/automl/neps/tree/master/neps_examples/basic_usage/yaml_usage)
## Parallelization

`neps.run` can be called multiple times with multiple processes or machines, to parallelize the optimization process.
Ensure that `root_directory` points to a shared location across all instances to synchronize the optimization efforts.
For more information [look here](https://automl.github.io/neps/latest/parallelization)

## Customization

The `neps.run` function allows for extensive customization through its arguments, enabling to adapt the
optimization process to the complexities of your specific problems.

For a deeper understanding of how to use `neps.run` in a practical scenario, take a look at our
[examples and templates](https://github.com/automl/neps/tree/master/neps_examples).
