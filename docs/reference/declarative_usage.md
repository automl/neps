!!! note "Work in Progress"
    This document is currently a work in progress and may contain incomplete or preliminary information.

## Introduction
### Configuring with YAML
Configure your experiments using a YAML file, which serves as a central reference for setting up your project.
This approach simplifies sharing, reproducing, and modifying configurations.
#### Simple YAML Example
Below is a straightforward YAML configuration example for NePS covering the required arguments.
=== "config.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/simple_example.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    def run_pipeline(learning_rate, optimizer, epochs):
        pass
    neps.run(run_pipeline, run_args="path/to/your/config.yaml")
    ```


#### Advanced Configuration with External Pipeline
In addition to setting experimental parameters via YAML, this configuration example also specifies the pipeline function
and its location, enabling more flexible project structures.
=== "config.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/simple_example_including_run_pipeline.yaml"
    ```
=== "run_neps.py"
    ```python
    import neps
    # No need to define run_pipeline here. NePS loads it directly from the specified path.
    neps.run(run_args="path/to/your/config.yaml")
    ```

#### Extended Configuration
This example showcases a more comprehensive YAML configuration, which includes not only the essential parameters
but also advanced settings for more complex setups:
=== "config.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/full_configuration_template.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    # Executes the configuration specified in your YAML file
    neps.run(run_args="path/to/your/config.yaml")
    ```

The `searcher` key used in the YAML configuration corresponds to the same keys used for selecting an optimizer directly
through `neps.run`. For a detailed list of integrated optimizers, see [here](optimizers.md#list-available-searchers)
!!! note "Note on Undefined Keys"
    Not all configurations are explicitly defined in this template. Any undefined key in the YAML file is mapped to
    the internal default settings of NePS. This ensures that your experiments can run even if certain parameters are
    omitted.

## Different Use Cases
### Customizing neps optimizer
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/customizing_neps_optimizer.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

# TODO
link here optimizer page, regarding arguments of optimizers


### Load your own optimizer
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/loading_own_optimizer.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### How to define hooks?
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/defining_hooks.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### What if your search space is big?
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/outsourcing_pipeline_space.yaml"
    ```

=== "pipeline_space.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/pipeline_space.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### If you experimenting a lot with different optimizer settings
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/outsourcing_optimizer.yaml"
    ```

=== "searcher_setup.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/set_up_optimizer.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### Architecture search space (Loading Dict)
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/loading_pipeline_space_dict.yaml"
    ```

=== "search_space.py"
    ```python
    search_space = {}
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### Multi-fidelity

### Prior-Band


