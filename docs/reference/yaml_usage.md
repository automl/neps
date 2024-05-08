!!! note "Work in Progress"
    This document is currently a work in progress and may contain incomplete or preliminary information.

# YAML Usage
For defining `run_args=`, some settings like `run_pipeline=`, `pre_load_hooks=`, and custom `searcher=` need both a path and a name to dynamically load them.
This is crucial for components where NePS must find and execute specific code.
Below, the configurations for dynamic loading are outlined:

- `run_pipeline=`:
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


For detailed examples and specific use case configurations, visit [here](../examples/yaml_usage/index.md)

