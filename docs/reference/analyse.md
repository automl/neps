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

## Summary CSV
The argument `post_run_summary` in `neps.run` allows for the automatic generation of CSV files after a run is complete.
The new root directory after utilizing this argument will look like the following:

```
ROOT_DIRECTORY
├── results
│  └── config_1
│      ├── config.yaml
│      ├── metadata.yaml
│      └── result.yaml
├── summary_csv
│  ├── config_data.csv
│  └── run_status.csv
├── all_losses_and_configs.txt
├── best_loss_trajectory.txt
└── best_loss_with_config_trajectory.txt
```

- *`config_data.csv`*: Contains all configuration details in CSV format, ordered by ascending `loss`.
Details include configuration hyperparameters, any returned result from the `run_pipeline` function, and metadata information.

- *`run_status.csv`*: Provides general run details, such as the number of sampled configs, best configs, number of failed configs, best loss, etc.

## TensorBoard Integration

### Introduction

[TensorBoard](https://www.tensorflow.org/tensorboard) serves as a valuable tool for visualizing machine learning experiments, offering the ability to observe losses and metrics throughout the model training process.
In NePS, we use this powerful tool to show metrics of configurations during training in addition to comparisons to different hyperparameters used in the search for better diagnosis of the model.

### The Logging Function

The `tblogger.log` function is invoked within the model's training loop to facilitate logging of key metrics.

!!! tip

    The logger function is primarily designed for implementation within the `run_pipeline` function during the training of the neural network.

- **Signature:**
```python
tblogger.log(
    loss: float,
    current_epoch: int,
    write_config_scalar: bool = False,
    write_config_hparam: bool = True,
    write_summary_incumbent: bool = False,
    extra_data: dict | None = None
)
```

- **Parameters:**
    - `loss` (float): The loss value to be logged.
    - `current_epoch` (int): The current epoch or iteration number.
    - `write_config_scalar` (bool, optional): Set to `True` for a live loss trajectory for each configuration.
    - `write_config_hparam` (bool, optional): Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
    - `write_summary_incumbent` (bool, optional): Set to `True` for a live incumbent trajectory.
    - `extra_data` (dict, optional): Additional data to be logged, provided as a dictionary.

### Extra Custom Logging

NePS provides dedicated functions for customized logging using the `extra_data` argument.

!!! note "Custom Logging Instructions"

    Name the dictionary keys as the names of the values you want to log and pass one of the following functions as the values for a successful logging process.

#### 1- Extra Scalar Logging

Logs new scalar data during training. Uses `current_epoch` from the log function as its `global_step`.

- **Signature:**
```python
tblogger.scalar_logging(value: float)
```
- **Parameters:**
    - `value` (float): Any scalar value to be logged at the current epoch of `tblogger.log` function.

#### 2- Extra Image Logging

Logs images during training. Images can be resized, randomly selected, and a specified number can be logged at specified intervals. Uses `current_epoch` from the log function as its `global_step`.

- **Signature:**
```python
tblogger.image_logging(
    image: torch.Tensor,
    counter: int = 1,
    resize_images: list[None | int] | None = None,
    random_images: bool = True,
    num_images: int = 20,
    seed: int | np.random.RandomState | None = None,
)
```

- **Parameters:**
    - `image` (torch.Tensor): Image tensor to be logged.
    - `counter` (int): Log images every counter epochs (i.e., when current_epoch % counter equals 0).
    - `resize_images` (list of int, optional): List of integers for image sizes after resizing (default: [32, 32]).
    - `random_images` (bool, optional): Images are randomly selected if True (default: True).
    - `num_images` (int, optional): Number of images to log (default: 20).
    - `seed` (int or np.random.RandomState or None, optional): Seed value or RandomState instance to control randomness and reproducibility (default: None).

### Logging Example

For illustration purposes, we have employed a straightforward example involving the tuning of hyperparameters for a model utilized in the classification of the MNIST dataset provided by [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

You can find this example [here](../examples/convenience/neps_tblogger_tutorial.md)

!!! info "Important"

    We have optimized the example for computational efficiency. If you wish to replicate the exact results showcased in the following section, we recommend the following modifications:

    1- Increase maximum epochs from 2 to 10

    2- Set the `write_summary_incumbent` argument to `True`

    3- Change the searcher from `random_search` to `bayesian_optimization`

    4- Increase the maximum evaluations before disabling `tblogger` from 2 to 14

    5- Increase the maximum evaluations after disabling `tblogger` from 3 to 15

### Visualization Results

The following command will open a local host for TensorBoard visualizations, allowing you to view them either in real-time or after the run is complete.

```bash
tensorboard --logdir path/to/root_directory
```

This image shows visualizations related to scalar values logged during training. Scalars typically include metrics such as loss, incumbent trajectory, a summary of losses for all configurations, and any additional data provided via the `extra_data` argument in the `tblogger.log` function.

![scalar_loggings](../doc_images/tensorboard/tblogger_scalar.jpg)

This image represents visualizations related to logged images during training.
It could include snapshots of input data, model predictions, or any other image-related information.
In our case, we use images to depict instances of incorrect predictions made by the model.

![image_loggings](../doc_images/tensorboard/tblogger_image.jpg)

The following images showcase visualizations related to hyperparameter logging in TensorBoard.
These plots include three different views, providing insights into the relationship between different hyperparameters and their impact on the model.

In the table view, you can explore hyperparameter configurations across five different trials.
The table displays various hyperparameter values alongside corresponding evaluation metrics.

![hparam_loggings1](../doc_images/tensorboard/tblogger_hparam1.jpg)

The parallel coordinate plot offers a holistic perspective on hyperparameter configurations.
By presenting multiple hyperparameters simultaneously, this view allows you to observe the interactions between variables, providing insights into their combined influence on the model.

![hparam_loggings2](../doc_images/tensorboard/tblogger_hparam2.jpg)

The scatter plot matrix view provides an in-depth analysis of pairwise relationships between different hyperparameters.
By visualizing correlations and patterns, this view aids in identifying key interactions that may influence the model's performance.

![hparam_loggings3](../doc_images/tensorboard/tblogger_hparam3.jpg)

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

## CLI commands

To generate plots to the root directory, run

```bash
python -m neps.plot ROOT_DIRECTORY
```

Currently, this creates one plot that shows the best error value across the number of evaluations.
