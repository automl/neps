# Parallelization and Resuming Runs

NePS utilizes files as a means of communication for implementing parallelization and resuming runs. As a result,
when `neps.run` is called multiple times with the same `root_directory` in the file system, NePS will automatically
load the optimizer state, allowing seamless parallelization of the run across different processes or machines.
This concept also applies to resuming runs even after termination.

Example:

!!! note
    The following example assumes all necessary imports are included, in addition to already having defined the [pipeline_space](https://automl.github.io/neps/latest/pipeline_space/) and the [run_pipeline](https://automl.github.io/neps/latest/run_pipeline/) functions. One can apply the same idea on [this](https://github.com/automl/neps/blob/master/neps_examples/basic_usage/hyperparameters.py) example.

```python
logging.basicConfig(level=logging.INFO)

# Initial run
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/my_example",
    max_evaluations_total=5,
)
```

After the initial run, NePS will log the following message:

```bash
INFO:neps:Maximum total evaluations is reached, shutting down
```

If you wish to extend the search with more evaluations, simply update the `max_evaluations_total` parameter:

```python
logging.basicConfig(level=logging.INFO)


# Resuming run with increased evaluations
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/my_example",
    max_evaluations_total=10,
)
```

Now, NePS will continue the search, loading the latest information for the searcher. For parallelization, as mentioned above, you can also run this code multiple times on different processes or machines. The file system communication will link them, as long as the `root_directory` has the same location.
