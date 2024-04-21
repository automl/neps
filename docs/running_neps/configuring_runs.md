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
