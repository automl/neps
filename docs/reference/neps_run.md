# Configuring and Running Optimizations

The [`neps.run()`][neps.api.run] function is the core interface for running Hyperparameter
and/or architecture search using optimizers in NePS.
You can find most of the features NePS provides through the API of this function.

This document breaks down the core arguments that allow users to control the optimization process in NePS.

---

## Required Arguments
To operate, NePS requires at minimum the following two arguments
[`neps.run(evaluate_pipeline=..., pipeline_space=...)`][neps.api.run]:

```python
import neps

def evaluate_pipeline(learning_rate: float, epochs: int) -> float:
    # Your code here

    return loss

neps.run(
    evaluate_pipeline=evaluate_pipeline, # (1)!
    pipeline_space={, # (2)!
        "learning_rate": neps.Float(1e-3, 1e-1, log=True),
        "epochs": neps.Integer(10, 100)
    },
    root_directory="path/to/result_dir" # (3)!
)
```

1.  The objective function, targeted by NePS for minimization, by evaluation various configurations.
    It requires these configurations as input and should return either a dictionary or a sole loss value as the output.
2.  This defines the search space for the configurations from which the optimizer samples.
    It accepts either a dictionary with the configuration names as keys, a path to a YAML configuration file, or a [`configSpace.ConfigurationSpace`](https://automl.github.io/ConfigSpace/) object.
    For comprehensive information and examples, please refer to the detailed guide available [here](../reference/pipeline_space.md)
3.  The directory path where the information about the optimization and its progress gets stored.
    This is also used to synchronize multiple calls to `neps.run()` for parallelization.


See the following for more:

* What kind of [pipeline space](../reference/pipeline_space.md) can you define?
* What goes in and what goes out of [`evaluate_pipeline()`](../reference/evaluate_pipeline.md)?

## Budget, how long to run?
To define a budget, provide `evaluations_to_spend=` to [`neps.run()`][neps.api.run],
to specify the total number of evaluations to conduct before halting the optimization process,
or `cost_to_spend=` to specify a cost threshold for your own custom cost metric, such as time, energy, or monetary, as returned by each evaluation of the pipeline .


```python
def evaluate_pipeline(learning_rate: float, epochs: int) -> float:
    start = time.time()

    # Your code here
    end = time.time()
    duration = end - start
    return {"objective_function_to_minimize": loss, "cost": duration}

neps.run(
    evaluations_to_spend=10, # (1)!
    cost_to_spend=1000, # (2)!
)
```

1.  Specifies the total number of evaluations to conduct before halting the optimization process.
2.  Prevents the initiation of new evaluations once this cost threshold is surpassed.
    This can be any kind of cost metric you like, such as time, energy, or monetary, as long as you can calculate it.
    This requires adding a cost value to the output of the `evaluate_pipeline` function, for example, return `#!python {'objective_to_minimize': loss, 'cost': cost}`.
    For more details, please refer [here](../reference/evaluate_pipeline.md)

## Getting some feedback, logging
NePS will not print anything to the console. To view the progress of workers,
you can enable logging through python's [logging.basicConfig][].

```python
import logging

logging.basicConfig(level=logging.INFO)

neps.run(...)
```

Please refer to Python's [logging documentation](https://docs.python.org/3/library/logging.html) for more information on how to customize the logging output.

## Continuing Runs
To continue a run, all you need to do is provide the same `root_directory=` to [`neps.run()`][neps.api.run] as before,
with an increased `evaluations_to_spend=` or `cost_to_spend=`.

```python
def run(learning_rate: float, epochs: int) -> float:
    start = time.time()

    # Your code here
    end = time.time()
    duration = end - start
    return {"objective_to_minimize": loss, "cost": duration}

neps.run(
    # Increase the total number of trials from 10 as set previously to 50
    evaluations_to_spend=50,
)
```

If the run previously stopped due to reaching a budget and you specify the same budget, the worker will immediatly stop as it will remember the amount of budget it used previously.

## Overwriting a Run

To overwrite a run, simply provide the same `root_directory=` to [`neps.run()`][neps.api.run] as before, with the `overwrite_root_directory=True` argument.

```python
neps.run(
    ...,
    root_directory="path/to/previous_result_dir",
    overwrite_root_directory=True,
)
```

!!! warning

    This will delete the folder specified by `root_directory=` and all its contents.

## Getting the results
The results of the optimization process are stored in the `root_directory=`
provided to [`neps.run()`][neps.api.run].

=== "Result Directory"

    The root directory after utilizing this argument will look like the following:

    ```
    root_directory
    ├── configs
    │   ├── config_1
    │   │   ├── config.yaml     # The configuration
    │   │   ├── report.yaml     # The results of this run, if any
    │   │   └── metadata.json   # Metadata about this run, such as state and times
    │   └── config_2
    │       ├── config.yaml
    │       └── metadata.json
    ├── summary                 
    │  ├── full.csv
    │  └── short.csv
    │  ├── best_config_trajectory.txt
    │  └── best_config.txt
    ├── optimizer_info.yaml     # The optimizer's configuration
    └── optimizer_state.pkl     # The optimizer's state, shared between workers
    ```

To capture the results of the optimization process, you can use tensorbaord logging with various utilities to integrate
closer to NePS. For more information, please refer to the [analyses page](../reference/analyse.md) page.

## Parallelization

NePS utilizes the file-system and locks as a means of communication for implementing parallelization and resuming runs.
As a result, you can start multiple [`neps.run()`][neps.api.run] from different processes however you like
and they will synchronize, **as long as they share the same `root_directory=`**.
Any new workers that come online will automatically pick up work and work together to until the budget is exhausted.

=== "Worker script"

    ```python
    # worker.py
    neps.run(
        evaluate_pipeline=...,
        pipeline_space=...,
        root_directory="some/path",
        evaluations_to_spend=100, # (1)!
        continue_until_max_evaluation_completed=True, # (2)!
        overwrite_root_directory=False, #!!!
    )
    ```

    1.  Limits the number of evaluations for this specific call of [`neps.run()`][neps.api.run].
    2.  Evaluations in-progress count towards evaluations_to_spend, halting new ones when this limit is reached.
        Setting this to `True` enables continuous sampling of new evaluations until the total of completed ones meets evaluations_to_spend, optimizing resource use in time-sensitive scenarios.

    !!! warning

        Ensure `overwrite_root_directory=False` to prevent newly spawned workers from deleting the shared directory!


=== "Shell"

    ```bash
    # Start 3 workers
    python worker.py &
    python worker.py &
    python worker.py &
    ```

## Handling Errors

Things go wrong during optimization runs and it's important to consider what to do in these cases.
By default, NePS will halt the optimization process when an error but you can choose to `ignore_errors=`,
providing a `loss_value_on_error=` and `cost_value_on_error=` to control what values should be
reported to the optimization process.

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

    Any runs that error will still count towards the total `evaluations_to_spend`.

### Re-running Failed Configurations

Sometimes things go wrong but not due to the configuration itself.
Sometimes you'd also like to change the state so that you re-evaluate that configuration.

If you need to go in there and change anything, **the entire optimization state** is editable on disk.
You can follow these steps to modify the state of things.

``` python
root_directory
├── configs
│   ├── .trial_cache.pkl    # A cache of all trial information for optimizers
│   ├── config_1
│   │   ├── config.yaml     # The configuration
│   │   ├── report.yaml     # The results of this run, if any
│   │   ├── metadata.json   # Metadata about this run, such as state and times
│   └── config_2
│       ├── config.yaml
│       └── metadata.json
├── optimizer_info.yaml
└── optimizer_state.pkl     # The optimizer's state, shared between workers
```

1. The first thing you should do is make sure no workers are running.
2. Next, delete `optimizer_state.pkl` and `configs/.trial_cache.pkl`. This is cached information to share between the
   workers.
3. Lastly, you can go in and modify any of the following files:

    * `config.yaml` - The configuration to be run. This was sampled from your search space.
    * `report.yaml` - The results of the run. This is where you can change what was reported back.
    * `metadata.json` - Metadata about the run. Here you can change the `"state"` key to one
        of [`State`][neps.state.trial.State] to re-run the configuration, usually you'd want to set it
        to `"pending"` such that the next worker will pick it up and re-run it.
4. Once you've made your changes, you can start the workers again and they will pick up the new state
    re-creating the caches as necessary.

## Selecting an Optimizer
By default NePS intelligently selects the most appropriate optimizer based on your defined configurations in `pipeline_space=`, one of the arguments to [`neps.run()`][neps.api.run].

The characteristics of your search space, as represented in the `pipeline_space=`, play a crucial role in determining which optimizer NePS will choose.
This automatic selection process ensures that the optimizer aligns with the specific requirements and nuances of your search space, thereby optimizing the effectiveness of the hyperparameter and/or architecture optimization.

You can also manually select a specific or custom optimizer that better matches your specific needs.
For more information about the available optimizers and how to customize your own, refer [here](../reference/optimizers.md).
