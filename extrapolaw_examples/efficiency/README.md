# Parallelization

In order to run extrapolaw in parallel on multiple processes or multiple machines, simply call `extrapolaw.acquire` multiple times.
All calls to `extrapolaw.acquire` need to use the same `root_directory` on the same filesystem to synchronize between the `extrapolaw.acquire`'s.

For example, start the HPO example in two shells from the same directory as below.

In shell 1:

```bash
python -m extrapolaw_examples.basic_usage.1_hyperparameters
```

In shell 2:

```bash
python -m extrapolaw_examples.basic_usage.1_hyperparameters
```
