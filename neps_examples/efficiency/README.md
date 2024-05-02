# Parallelization

In order to run neps in parallel on multiple processes or multiple machines, simply call `neps.run` multiple times.
All calls to `neps.run` need to use the same `root_directory` on the same filesystem to synchronize between the `neps.run`'s.

For example, start the HPO example in two shells from the same directory as below.

In shell 1:

```bash
python -m neps_examples.basic_usage.hyperparameters
```

In shell 2:

```bash
python -m neps_examples.basic_usage.hyperparameters
```
