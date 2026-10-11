# Parallelization

In order to run autoscan in parallel on multiple processes or multiple machines, simply call `autoscan.run` multiple times.
All calls to `autoscan.run` need to use the same `root_directory` on the same filesystem to synchronize between the `autoscan.run`'s.

For example, start the HPO example in two shells from the same directory as below.

In shell 1:

```bash
python -m autoscan_examples.basic_usage.hyperparameters
```

In shell 2:

```bash
python -m autoscan_examples.basic_usage.hyperparameters
```
