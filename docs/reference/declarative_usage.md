
## Introduction
### Configuring with YAML
Configure your experiments using a YAML file, which serves as a central reference for setting up your project.
This approach simplifies sharing, reproducing and modifying configurations.

!!! note "Argument Handling and Prioritization"
    You can partially define and provide arguments via `run_args` (YAML file) and partially provide the arguments
    directly to `neps.run`. Arguments directly provided to `neps.run` get prioritized over those defined in the YAML file. An exception to this
    is for `searcher_kwargs` where a merge happens between the configurations. In this case, the directly provided arguments
    are still prioritized, but the values from both the directly provided arguments and the YAML file are merged.


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
        model = initialize_model()
        training_loss = train_model(model, optimizer, learning_rate, epochs)
        evaluation_loss = evaluate_model(model)
        return {"loss": evaluation_loss, "training_loss": training_loss}

    if __name__ == "__main__":
        neps.run(run_pipeline, run_args="path/to/your/config.yaml")
    ```


#### Including `run_pipeline` in `run_args` for External Referencing
In addition to setting experimental parameters via YAML, this configuration example also specifies the pipeline function
and its location, enabling more flexible project structures.
=== "config.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/simple_example_including_run_pipeline.yaml"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline_extended.py"
    ```


#### Comprehensive YAML Configuration Template
This example showcases a more comprehensive YAML configuration, which includes not only the essential parameters
but also advanced settings for more complex setups.
=== "config.yaml"
    ```yaml
    --8<-- "docs/doc_yamls/full_configuration_template.yaml"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline_extended.py"
    ```

The `searcher` key used in the YAML configuration corresponds to the same keys used for selecting an optimizer directly
through `neps.run`. For a detailed list of integrated optimizers, see [here](optimizers.md#list-available-searchers)
!!! note "Note on undefined keys in `run_args` (config.yaml)"
    Not all configurations are explicitly defined in this template. Any undefined key in the YAML file is mapped to
    the internal default settings of NePS. This ensures that your experiments can run even if certain parameters are
    omitted.

## Different Use Cases
### Customizing NePS optimizer
Customize an internal NePS optimizer by specifying its parameters directly under the key `searcher` in the
`config.yaml` file.

!!! note
    For `searcher_kwargs` of `neps.run`, the optimizer arguments passed via the YAML file and those passed directly via
    `neps.run` will be merged. In this special case, if the same argument is referenced in both places,
    `searcher_kwargs` will be prioritized and set for this argument.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/customizing_neps_optimizer.yaml"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline.py"
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
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline.py"
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
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline_big_search_space.py"
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
    --8<-- "docs/doc_yamls/architecture_search_space.py"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline_architecture.py"
    ```


### Integrating Custom Optimizers
For people who want to write their own optimizer class as a subclass of the base optimizer, you can load your own
custom optimizer class and define its arguments in `config.yaml`.

Note: You can still overwrite arguments via searcher_kwargs of `neps.run` like for the internal searchers.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/loading_own_optimizer.yaml"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline.py"
    ```

### Adding Custom Hooks to Your Configuration
Define hooks in your YAML configuration to extend the functionality of your experiment.
=== "config.yaml"
    ```yaml
        --8<-- "docs/doc_yamls/defining_hooks.yaml"
    ```
=== "run_pipeline.py"
    ```python
    --8<-- "docs/doc_yamls/run_pipeline_extended.py"
    ```

=== "run_neps.py"
    ```python
    import neps
    neps.run(run_args="path/to/your/config.yaml")
    ```

## CLI Usage
This section provides a brief overview of the primary commands available in the NePS CLI.
For additional command options, you can directly refer to the help documentation
provided by each command using --help.


### **`init` Command**

Generates a default `run_args` YAML configuration file, providing a template that you can customize for your experiments.

**Options:**

  - `--config-path <path>`: *Optional*. Specify the custom path for generating the configuration file. Defaults to
  `run_config.yaml` in the current working directory.
  - `--template [basic|complete]`: *Optional*. Choose between a basic or complete template. The basic template includes only required settings, while the complete template includes all NePS configurations.
  - `--state-machine`: *Optional*. Creates a NEPS state if set, which requires an existing `run_config.yaml`.

**Example Usage:**
```bash
neps init --config-path custom/path/config.yaml --template complete
```

### **`run` Command**

Executes the optimization based on the provided configuration. This command serves as a CLI wrapper around `neps.run`, effectively mapping each CLI argument to a parameter in `neps.run`. It offers a flexible interface that allows you to override the existing settings specified in the YAML configuration file, facilitating dynamic adjustments for managing your experiments.

**Options:**

  - `--run-args <path>`: Path to the YAML configuration file containing the run arguments.
  - `--run-pipeline <path/to/module:function_name>`: *Optional*. Specify the path to the Python module and function to use for running the pipeline. Overrides any settings in the YAML file.
  - `--pipeline-space <path/to/yaml>`: Path to the YAML file defining the search space for the optimization.
  - `--root-directory <path>`: *Optional*. Directory for saving progress and synchronizing multiple processes. Defaults to the `root_directory` from `run_config.yaml` if not provided.
  - `--overwrite-working-directory`: *Optional*. If set, deletes the working directory at the start of the run.
  - `--development-stage-id <id>`: *Optional*. Identifier for the current development stage, useful for multi-stage projects.
  - `--task-id <id>`: *Optional*. Identifier for the current task, useful for managing projects with multiple tasks.
  - `--post-run-summary/--no-post-run-summary`: *Optional*. Provides a summary of the run after execution. Enabled by default.
  - `--max-evaluations-total <int>`: *Optional*. Specifies the total number of evaluations to run.
  - `--max-evaluations-per-run <int>`: *Optional*. Number of evaluations to run per call.
  - `--continue-until-max-evaluation-completed`: *Optional*. If set, ensures the run continues until `max-evaluations-total` has been reached.
  - `--max-cost-total <float>`: *Optional*. Specifies a cost threshold. No new evaluations will start if this cost is exceeded.
  - `--ignore-errors`: *Optional*. If set, errors during the optimization will be ignored.
  - `--loss-value-on-error <float>`: *Optional*. Specifies the loss value to assume in case of an error.
  - `--cost-value-on-error <float>`: *Optional*. Specifies the cost value to assume in case of an error.
  - `--searcher <key>`: Specifies the searcher algorithm for optimization.
  - `--searcher-kwargs <key=value>`: *Optional*. Additional keyword arguments for the searcher.

**Example Usage:**
```bash
neps run --run-args path/to/config.yaml --max-evaluations-total 50
```

### **`status` Command**
Executes the optimization based on the provided configuration. This command serves as a CLI wrapper around neps.run,
effectively mapping each CLI argument to a parameter in neps.run. This setup offers a flexible interface that allows
you to override the existing settings specified in the YAML configuration file, facilitating dynamic adjustments for
managing your experiments.

**Example Usage:**
```bash
neps run --run-args path/to/config.yaml
```