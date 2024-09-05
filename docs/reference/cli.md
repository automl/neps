# NePS Command Line Interface
This section provides a brief overview of the primary commands available in the NePS CLI.

## **`init` Command**

Generates a default `run_args` YAML configuration file, providing a template that you can customize for your experiments.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--config-path` (Optional): Optional custom path for generating the configuration file. Default is 'config.yaml'.
- `--template` (Optional): Optional, options between different templates. Required configs(basic) vs all neps configs (complete)
- `--state-machine` (Optional): If set, creates a NEPS state. Requires an existing config.yaml.


**Example Usage:**

```bash
neps init --config-path custom/path/config.yaml --template complete
```


## **`run` Command**

Executes the optimization based on the provided configuration. This command serves as a CLI wrapper around `neps.run`, effectively mapping each CLI argument to a parameter in `neps.run`. It offers a flexible interface that allows you to override the existing settings specified in the YAML configuration file, facilitating dynamic adjustments for managing your experiments.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--run-args` (Optional): Path to the YAML configuration file.
- `--run-pipeline` (Optional): Optional: Provide the path to a Python file and a function name separated by a colon, e.g., 'path/to/module.py:function_name'. If provided, it overrides the run_pipeline setting from the YAML configuration.
- `--pipeline-space` (Optional): Path to the YAML file defining the search space for the optimization. This can be provided here or defined within the 'run_args' YAML file.
- `--root-directory` (Optional): The directory to save progress to. This is also used to synchronize multiple calls for parallelization.
- `--overwrite-working-directory` (Optional): If set, deletes the working directory at the start of the run. This is useful, for example, when debugging a run_pipeline function.
- `--development-stage-id` (Optional): Identifier for the current development stage, used in multi-stage projects.
- `--task-id` (Optional): Identifier for the current task, useful in projects with multiple tasks.
- `--post-run-summary` (Optional): Provide a summary of the results after running.
- `--no-post-run-summary` (Optional): Do not provide a summary of the results after running.
- `--max-evaluations-total` (Optional): Total number of evaluations to run.
- `--max-evaluations-per-run` (Optional): Number of evaluations a specific call should maximally do.
- `--continue-until-max-evaluation-completed` (Optional): If set, only stop after max-evaluations-total have been completed. This is only relevant in the parallel setting.
- `--max-cost-total` (Optional): No new evaluations will start when this cost is exceeded. Requires returning a cost
  in the run_pipeline function.
- `--ignore-errors` (Optional): If set, ignore errors during the optimization process.
- `--loss-value-on-error` (Optional): Loss value to assume on error.
- `--cost-value-on-error` (Optional): Cost value to assume on error.
- `--searcher` (Optional): String key of searcher algorithm to use for optimization.
- `--searcher-kwargs` (Optional): Additional keyword arguments as key=value pairs for the searcher.


**Example Usage:**

```bash
neps run --run-args path/to/config.yaml --max-evaluations-total 50
```


## **`status` Command**
Check the status of the NePS run. This command provides a summary of trials, including pending, evaluating, succeeded, and failed trials. You can filter the trials displayed based on their state.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.
- `--pending` (Optional): Show only pending trials.
- `--evaluating` (Optional): Show only evaluating trials.
- `--succeeded` (Optional): Show only succeeded trials.


**Example Usage:**
```bash
neps status --root-directory path/to/directory --succeeded
```


## **`info-config` Command**
Provides detailed information about a specific configuration identified by its ID. This includes metadata, configuration values, and trial status.

**Arguments:**


- id (Required): The configuration ID to be used.


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.


**Example Usage:**
```bash
neps info-config 42 --root-directory path/to/directory
```

## **`results` Command**
Displays the results of the NePS run, listing all incumbent trials in reverse order (most recent first). Optionally,
you can plot the results to visualize the progression of incumbents over trials.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.
- `--plot` (Optional): Plot the results if set.


**Example Usage:**

```bash
neps results --root-directory path/to/directory --plot
```



## **`errors` Command**
Lists all errors found in the specified NePS run. This is useful for debugging or reviewing failed trials.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.


**Example Usage:**

```bash
neps errors --root-directory path/to/directory
```


## **`sample-config` Command**


**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.


**Example Usage:**

```bash
neps sample-config --help
```


## **`help` Command**
Displays help information for the NePS CLI, including a list of available commands and their descriptions.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit


**Example Usage:**

```bash
neps help --help
```

