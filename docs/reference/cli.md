# NePS Command Line Interface
This section provides a brief overview of the commands available in the NePS CLI.

---

## **`init` Command**

Generates a default `run_args` YAML configuration file, providing a template that you can customize for your experiments.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--config-path` (Optional): Optional custom path for generating the configuration file. Default is 'run_config.yaml'.
- `--template` (Optional): Optional, options between different templates. Required configs(basic) vs all neps configs (complete)
- `--database` (Optional):  If set, creates the NePS database. This is required if you want to sample and report configurations using only CLI commands. Requires an existing config.yaml.


**Example Usage:**

```bash
neps init --config-path custom/path/config.yaml --template complete
```

---
## **`run` Command**

Executes the optimization based on the provided configuration. This command serves as a CLI wrapper around `neps.run`, effectively mapping each CLI argument to a parameter in `neps.run`. It offers a flexible interface that allows you to override the existing settings specified in the YAML configuration file, facilitating dynamic adjustments for managing your experiments.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--run-args` (Optional): Path to the YAML configuration file.
- `--run-pipeline` (Optional): Optional: Provide the path to a Python file and a function name separated by a colon, e.g., 'path/to/module.py:function_name'. If provided, it overrides the run_pipeline setting from the YAML configuration.
- `--pipeline-space` (Optional): Path to the YAML file defining the search space for the optimization. This can be provided here or defined within the 'run_args' YAML file.
- `--root-directory` (Optional): The directory to save progress to. This is also used to synchronize multiple calls for parallelization.
- `--overwrite-working-directory` (Optional): If set, deletes the working directory at the start of the run. This is useful, for example, when debugging a run_pipeline function.
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
- `--optimizer` (Optional): String key of optimizer algorithm to use for optimization.
- `--optimizer-kwargs` (Optional): Additional keyword arguments as key=value pairs for the optimizer.


**Example Usage:**

```bash
neps run --run-args path/to/config.yaml --max-evaluations-total 50
```

---
## **`status` Command**
Check the status of the NePS run. This command provides a summary of trials, including pending, evaluating, succeeded, and failed trials. You can filter the trials displayed based on their state.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): The path to your root_directory. If not provided, it will be loaded from run_config.yaml.
- `--pending` (Optional): Show only pending trials.
- `--evaluating` (Optional): Show only evaluating trials.
- `--succeeded` (Optional): Show only succeeded trials.


**Example Usage:**
```bash
neps status --root-directory path/to/directory --succeeded
```

---
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

---
## **`results` Command**
Displays the results of the NePS run, listing all incumbent trials in reverse order (most recent first). Optionally,
you can plot the results to visualize the progression of incumbents over trials.  Additionally, you can dump all
trials or incumbent trials to a file in the specified format and plot the results to visualize the progression of
incumbents over trials.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.
- `--plot` (Optional): Plot the incumbents if set.
- `--dump-all-configs` (Optional): Dump all information about the trials to a file in the specified format (csv, json,
  parquet).
- `--dump-incumbents` (Optional): Dump only the information about the incumbent trials to a file in the specified
  format (csv, json, parquet).



**Example Usage:**

```bash
neps results --root-directory path/to/directory --plot
```

---
## **`errors` Command**
Lists all errors found in the specified NePS run. This is useful for debugging or reviewing failed trials.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit
- `--root-directory` (Optional): Optional: The path to your root_directory. If not provided, it will be loaded from run_config.yaml.


**Example Usage:**

```bash
neps errors --root-directory path/to/directory
```


---
## **`sample-config` Command**
The sample-config command allows users to generate new configurations based on the current state of the
NePS optimizer. This is particularly useful when you need to manually intervene in the sampling process, such
as allocating different computational resources to different configurations.

!!! note "Note"
    Before using the `sample-config` command, you need to initialize the database by running `neps init --database` if you haven't already executed `neps run`. Running `neps run` will also create a `NePsState`.

**Arguments:**

- `-h, --help` (Optional): show this help message and exit
- `--worker-id` (Optional): The worker ID for which the configuration is being sampled.
- `--run-args` (Optional): Path to the YAML configuration file. If not provided, it will search after run_config.yaml.
- `--number-of-configs` (Optional): Number of configurations to sample (default: 1).


**Example Usage:**


```bash
neps sample-config --worker-id worker_1 --number-of-configs 5
```

---
## **`report-config` Command**
The `report-config` command is the counterpart to `sample-config` and reports the outcome of a specific trial by updating its status and associated metrics in the NePS state. This command is crucial for manually managing the evaluation results of sampled configurations.

**Arguments:**


- `` (Required): ID of the trial to report
- `` (Required): Outcome of the trial


- `-h, --help` (Optional): show this help message and exit
- `--worker-id` (Optional): The worker ID for which the configuration is being sampled.
- `--loss` (Optional): Loss value of the trial
- `--run-args` (Optional): Path to the YAML file containing run configurations
- `--cost` (Optional): Cost value of the trial
- `--learning-curve` (Optional): Learning curve as a list of floats, provided like this --learning-curve 0.9 0.3 0.1
- `--duration` (Optional): Duration of the evaluation in sec
- `--err` (Optional): Error message if any
- `--tb` (Optional): Traceback information if any
- `--time-end` (Optional): The time the trial ended as either a UNIX timestamp (float) or in 'YYYY-MM-DD HH:MM:SS' format


**Example Usage:**


```bash
neps report-config 42 success --worker-id worker_1 --loss 0.95 --duration 120
```

---
## **`help` Command**
Displays help information for the NePS CLI, including a list of available commands and their descriptions.

**Arguments:**


- `-h, --help` (Optional): show this help message and exit


**Example Usage:**

```bash
neps help --help
```

---
## **Using NePS as a State Machine**

NePS can function as a state machine, allowing you to manually sample and report configurations using CLI commands. This is particularly useful in scenarios like architecture search, where different configurations may require varying computational resources. To utilize NePS in this manner, follow these steps:

### **Step 1: Initialize and Configure `run_config.yaml**

Begin by generating the `run_args` YAML configuration file. This file serves as the blueprint for your optimization experiments.


```bash
neps init
```
The `neps init` command creates run_config.yaml, which serves as the default configuration resource for all NePS commands.
### **Step 2: Initialize the NePS Database**

Set up the NePS database to enable the sampling and reporting of configurations via CLI commands.

```bash
neps init --database
```
This command initializes the NePS database, preparing the necessary folders and files required for managing your NePS run


### **Step 3: Sample Configurations**

Generate new configurations based on the existing NePS state. This step allows you to create configurations that you can manually evaluate.

```bash
neps sample-config --worker-id worker_1 --number-of-configs 5
```

- **`--worker_id worker_1`**: Identifies the worker responsible for sampling configurations.
- **`--number-of-configs 5`**: Specifies the number of configurations to sample.

### **Step 4: Evaluate and Report Configurations**

After evaluating each sampled configuration, report its outcome to update the NePS state.

```bash
neps report-config 42 success --worker-id worker_1 --loss 0.95 --duration 120
```

- **`42`**: The ID of the trial being reported.
- **`success`**: The outcome of the trial (`success`, `failed`, `crashed`).
- **`--worker_id worker_1`**: Identifies the worker reporting the configuration.
- **`--loss 0.95`**: The loss value obtained from the trial.
- **`--duration 120`**: The duration of the evaluation in seconds.
