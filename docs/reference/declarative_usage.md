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


#### Including `run_pipeline` in config.yaml for External Referencing
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

#### Comprehensive YAML Configuration Template
This example showcases a more comprehensive YAML configuration, which includes not only the essential parameters
but also advanced settings for more complex setups.
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
### Customizing NePS optimizer
Customize an internal NePS optimizer by specifying its parameters directly in the `config.yaml`.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/customizing_neps_optimizer.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

For detailed information about the available optimizers and their parameters, please visit the [optimizer page](optimizers.md#list-available-searching-algorithms)


### Testing Multiple Optimizer Configurations
Simplify experiments with multiple optimizer settings by outsourcing the optimizer configuration.
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

### Handling Large Search Spaces
Manage large search spaces by outsourcing the pipeline space configuration in a separate YAML file or for keeping track
of your experiments.
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


### Using Architecture Search Spaces
Since the option for defining the search space via YAML is limited to HPO, grammar-based search spaces or architecture
search spaces must be loaded via a dictionary, which is then referenced in the `config.yaml`.
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


### Integrating Custom Optimizers
You can also load your own custom optimizer and change its arguments in `config.yaml`.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/loading_own_optimizer.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

### Adding Custom Hooks to Your Configuration
Define hooks in your YAML configuration to extend the functionality of your experiment.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/defining_hooks.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```
