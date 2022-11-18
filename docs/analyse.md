# Analysing Runs

NePS has some convenient utilities to help you to understand the results of your run.

## Saved to disk

In the root directory, NePS maintains several files at all times that are human readable and can be useful

```
ROOT_DIRECTORY
├── results
│  └── config_1
│      ├── config.yaml
│      ├── metadata.yaml
│      └── result.yaml
├── all_losses_and_configs.txt
├── best_loss_trajectory.txt
└── best_loss_with_config_trajectory.txt
```

## Status

To show status information about a neural pipeline search run, use

```bash
python -m neps.status ROOT_DIRECTORY
```

If you need more status information than is printed per default (e.g., the best config over time), please have a look at

```bash
python -m neps.status --help
```

To show the status repeatedly, on unix systems you can use

```bash
watch --interval 30 python -m neps.status ROOT_DIRECTORY
```

## Visualizations

To generate plots to the root directory, run

```bash
python -m neps.plot ROOT_DIRECTORY
```

Currently, this creates one plot that shows the best error value across the number of evaluations.
