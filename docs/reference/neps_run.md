# Configuring and Running Optimizations

The [`neps.run()`][neps.api.run] function is the core interface for running Hyperparameter and/or architecture search using optimizers in NePS.

This document breaks down the core arguments that allow users to control the optimization process in NePS.
Please see the documentation of [`neps.run()`][neps.api.run] for a full list.


## Required Arguments
To operate, NePS requires at minimum the following three arguments
[`neps.run(run_pipeline=..., pipeline_space=..., root_directory=...)`][neps.api.run]:

```python
import neps

def run(learning_rate: float, epochs: int) -> float:
    # Your code here

    return loss

neps.run(
    run_pipeline=run, # (1)!
    pipeline_space={, # (2)!
        "learning_rate": neps.Float(1e-3, 1e-1, log=True),
        "epochs": neps.Integer(10, 100)
    },
    root_directory="path/to/result_dir" # (3)!
)
```

1.  The objective function, targeted by NePS for minimization, by evaluation various configurations.
    It requires these configurations as input and should return either a dictionary or a sole loss value as the output.
    For correct setup instructions, refer to the [run pipeline page](../reference/run_pipeline.md)
2.  This defines the search space for the configurations from which the optimizer samples.
    It accepts either a dictionary with the configuration names as keys, a path to a YAML configuration file, or a [`configSpace.ConfigurationSpace`](https://automl.github.io/ConfigSpace/) object.
    For comprehensive information and examples, please refer to the detailed guide available [here](../reference/pipeline_space.md)
3.  The directory path where the information about the optimization and its progress gets stored.
    This is also used to synchronize multiple calls to `neps.run()` for parallelization.

To learn more about the `run_pipeline` function and the `pipeline_space` configuration, please refer to the [run pipeline](../reference/run_pipeline.md) and [pipeline space](../reference/pipeline_space.md) pages.

## Budget, how long to run?
To define a budget, provide `max_evaluations_total=` to [`neps.run()`][neps.api.run], to specify the total number of evaluations to conduct before halting the optimization process,
or `max_cost_total=` to specify a cost threshold for your own custom cost metric, such as time, energy, or monetary.


```python

```python
def run(learning_rate: float, epochs: int) -> float:
    start = time.time()

    # Your code here
    end = time.time()
    duration = end - start
    return {"loss": loss, "cost": duration}

neps.run(
    max_evaluations_total=10, # (1)!
    max_cost_total=1000, # (2)!
)
```

1.  Specifies the total number of evaluations to conduct before halting the optimization process.
2.  Prevents the initiation of new evaluations once this cost threshold is surpassed.
    This can be any kind of cost metric you like, such as time, energy, or monetary, as long as you can calculate it.
    This requires adding a cost value to the output of the `run_pipeline` function, for example, return `#!python {'loss': loss, 'cost': cost}`.
    For more details, please refer [here](../reference/run_pipeline.md)

## Getting some feedback, logging
Most of NePS will not print anything to the console.
To view the progress of workers, you can do so by enabling logging through [logging.basicConfig][].

```python
import logging

logging.basicConfig(level=logging.INFO)

neps.run(...)
```

Please refer to Python's [logging documentation](https://docs.python.org/3/library/logging.html) for more information on how to customize the logging output.

## Continuing Runs
To continue a run, all you need to do is provide the same `root_directory=` to [`neps.run()`][neps.api.run] as before,
with an increased `max_evaluations_total=` or `max_cost_total=`.

```python
def run(learning_rate: float, epochs: int) -> float:
    start = time.time()

    # Your code here
    end = time.time()
    duration = end - start
    return {"loss": loss, "cost": duration}

neps.run(
    # Increase the total number of trials from 10 as set previously to 50
    max_evaluations_total=50,
)
```

If the run previously stopped due to reaching a budget and you specify the same budget, the worker will immediatly stop as it will remember the amount of budget it used previously.

## Overwriting a Run

To overwrite a run, simply provide the same `root_directory=` to [`neps.run()`][neps.api.run] as before, with the `overwrite_working_directory=True` argument.

```python
neps.run(
    ...,
    root_directory="path/to/previous_result_dir",
    overwrite_working_directory=True,
)
```

!!! warning

    This will delete the folder specified by `root_directory=` and all its contents.

## Getting the results
The results of the optimization process are stored in the `root_directory=` provided to [`neps.run()`][neps.api.run].
To obtain a summary of the optimization process, you can enable the `post_run_summary=True` argument in [`neps.run()`][neps.api.run], while will generate a summary csv after the run has finished.

=== "Result Directory"

    The root directory after utilizing this argument will look like the following:

    ```
    ROOT_DIRECTORY
    ├── results
    │  └── config_1
    │      ├── config.yaml
    │      ├── metadata.yaml
    │      └── result.yaml
    ├── summary_csv         # Only if post_run_summary=True
    │  ├── config_data.csv
    │  └── run_status.csv
    ├── all_losses_and_configs.txt
    ├── best_loss_trajectory.txt
    └── best_loss_with_config_trajectory.txt
    ```

=== "python"

    ```python
    neps.run(..., post_run_summary=True)
    ```

To capture the results of the optimization process, you can use tensorbaord logging with various utilities to integrate
closer to NePS. For more information, please refer to the [analyses page](../reference/analyse.md) page.

## Parallelization
NePS utilizes the file-system and locks as a means of communication for implementing parallelization and resuming runs.
As a result, you can start multiple [`neps.run()`][neps.api.run] from different processes however you like and they will synchronize, **as long as they share the same `root_directory=`**.
Any new workers that come online will automatically pick up work and work together to until the budget is exhausted.

=== "Worker script"

    ```python
    # worker.py
    neps.run(
        run_pipeline=...,
        pipeline_space=...,
        root_directory="some/path",
        max_evaluations_total=100,
        max_evaluations_per_run=10, # (1)!
        continue_until_max_evaluation_completed=True, # (2)!
        overwrite_working_directory=False, #!!!
    )
    ```

    1.  Limits the number of evaluations for this specific call of [`neps.run()`][neps.api.run].
    2.  Evaluations in-progress count towards max_evaluations_total, halting new ones when this limit is reached.
        Setting this to `True` enables continuous sampling of new evaluations until the total of completed ones meets max_evaluations_total, optimizing resource use in time-sensitive scenarios.

    !!! warning

        Ensure `overwrite_working_directory=False` to prevent newly spawned workers from deleting the shared directory!


=== "Shell"

    ```bash
    # Start 3 workers
    python worker.py &
    python worker.py &
    python worker.py &
    ```

## YAML Configuration
We support arguments to [`neps.run()`][neps.api.run] that have been seriliazed into a
YAML file. This means you can manage your configurations in a more human-readable format
if you prefer.

For more on yaml usage, please visit the dedicated
[page on usage of YAML with NePS](../reference/declarative_usage.md).


=== "Yaml Configuration"

    ```yaml
    # path/to/your/config.yaml
    evaluate_pipeline: path/to/your/run_pipeline.py:name_of_your_run_pipeline_function

    pipeline_space:
      batch_size: 64                # Constant
      optimizer: [adam, sgd, adamw] # Categorical
      alpha: [0.01, 1.0]            # Uniform Float
      n_layers: [1, 10]             # Uniform Integer
      learning_rate:                # Log scale Float with a prior
        lower: 1e-5
        upper: 1e-1
        log: true
        prior: 1e-3
        prior_confidence: high
      epochs:                       # Integer fidelity
        lower: 5
        upper: 20
        is_fidelity: true

    root_directory: "neps_results"  # Output directory for results
    max_evaluations_total: 100
    optimizer:
      name: "bayesian_optimization"
      initial_design_size: 5
      surrogate_model: "gp"
    ```

=== "Python"

    ```python
    with open("path/to/your/config.yaml", "r") as file:
        settings = yaml.safe_load(file)

    neps.run(**settings)
    ```

## Handling Errors
Things go wrong during optimization runs and it's important to consider what to do in these cases.
By default, NePS will halt the optimization process when an error but you can choose to `ignore_errors=`, providing a `loss_value_on_error=` and `cost_value_on_error=` to control what values should be reported to the optimization process.

```python
def run(learning_rate: float, epochs: int) -> float:
    if whoops_my_gpu_died():
        raise RuntimeError("Oh no! GPU died!")

    ...
    return loss

neps.run(
    loss_value_on_error=100, # (1)!
    cost_value_on_error=10, # (2)!
    ignore_errors=True, # (3)!
)
```

1. If an error occurs, the **loss** value for that configuration will be set to 100.
2. If an error occurs, the **cost** value for that configuration will be set to 100.
3. Continue the optimization process even if an error occurs, otherwise throwing an exception and halting the process.

!!! note

    Any runs that error will still count towards the total `max_evaluations_total` or `max_evaluations_per_run`.

## Selecting an Optimizer
By default NePS intelligently selects the most appropriate optimizer based on your defined configurations in `pipeline_space=`, one of the arguments to [`neps.run()`][neps.api.run].

The characteristics of your search space, as represented in the `pipeline_space=`, play a crucial role in determining which optimizer NePS will choose.
This automatic selection process ensures that the optimizer aligns with the specific requirements and nuances of your search space, thereby optimizing the effectiveness of the hyperparameter and/or architecture optimization.

You can also manually select a specific or custom optimizer that better matches your specific needs.
For more information about the available optimizers and how to customize your own, refer [here](../reference/optimizers.md).
