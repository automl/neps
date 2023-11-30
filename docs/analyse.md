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

### 1. CLI commands

To generate plots to the root directory, run

```bash
python -m neps.plot ROOT_DIRECTORY
```

Currently, this creates one plot that shows the best error value across the number of evaluations.

### 2. TensorBoard

#### Introduction

[TensorBoard](https://www.tensorflow.org/tensorboard) serves as a valuable tool for visualizing machine learning experiments, offering the ability to observe losses and metrics throughout the model training process. In NePS, we use this powerful tool to show metrics of configurations during training in addition to comparisons to different hyperparameters used in the search for better diagnosis of the model.

#### The Logging Function

The `tblogger.log` function is invoked within the model's training loop to facilitate logging of key metrics.

- **Signature:**
```python
tblogger.log(
    loss: float,
    current_epoch: int,
    write_summary_incumbent: bool = False,
    write_config_scalar: bool = False,
    write_config_hparam: bool = True,
    extra_data: dict | None = None
)
```

- **Parameters:**
    - `loss` (float): The loss value to be logged.
    - `current_epoch` (int): The current epoch or iteration number.
    - `write_summary_incumbent` (bool, optional): Set to `True` for a live incumbent trajectory.
    - `write_config_scalar` (bool, optional): Set to `True` for a live loss trajectory for each configuration.
    - `write_config_hparam` (bool, optional): Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
    - `extra_data` (dict, optional): Additional data to be logged, provided as a dictionary.

#### Extra Custom Logging

NePS provides dedicated functions for customized logging using the `extra_data` argument. 

!!! note "Custom Logging Instructions"

    Name the dictionary keys as the names of the values you want to log and pass one of the following functions as the values for a successful logging process.

##### 1- Extra Scalar Logging

Logs new scalar data during training. Uses `current_epoch` from the log function as its `global_step`.

- **Signature:**
```python
tblogger.scalar_logging(value: float)
```
- **Parameters:**
    - `value` (float): Any scalar value to be logged at the current epoch of `tblogger.log` function.

##### 2- Extra Image Logging

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

#### Logging Example

For illustration purposes, we will employ a straightforward example involving the tuning of hyperparameters for a model utilized in the classification of the MNIST dataset provided by [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

To begin, we'll set up the required imports, establish a seed for reproducibility, and download the necessary training and testing datasets.

```py linenums="1"
import logging
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

import neps
from neps.plot.tensorboard_eval import tblogger


# Definig the seeds for reproducibility
def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Prepare the input data. For this tutorial we use the MNIST dataset.
def MNIST(
    batch_size: int = 128,
    data_reduction_factor: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    # Download MNIST training and test datasets if not already downloaded.
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Determine the size of the reduced training dataset for faster training
    reduced_train_size = int(data_reduction_factor * len(train_dataset))
    reduced_test_size = int(data_reduction_factor * len(test_dataset))

    # Create a subset of the training dataset based on the reduction factor
    reduced_train_dataset = torch.utils.data.Subset(
        train_dataset, range(reduced_train_size)
    )
    reduced_test_dataset = torch.utils.data.Subset(
        test_dataset, range(reduced_test_size)
    )

    # Create DataLoaders for training, validation, and test datasets.
    train_dataloader = DataLoader(
        dataset=reduced_train_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=reduced_test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader


# Designing a model to be able to represent the input data.
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=196)
        self.linear2 = nn.Linear(in_features=196, out_features=98)
        self.linear3 = nn.Linear(in_features=98, out_features=10)

    def forward(self, x: torch.Tensor):
        # Flattening the grayscaled image from 1x28x28 (CxWxH) to 784.
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

```

Now, we begin constructing our hyperparameter search algorithm with NePS, guided by the three fundamental pillars: the pipeline space, the run_pipeline function, and the neps.run command.

```py linenums="1" hl_lines="55-72"
# Desinging the pipeline space
def pipeline_space() -> dict:
    search_space = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
        batch_size=neps.IntegerParameter(lower=32, upper=256),
        weight_decay=neps.FloatParameter(lower=1e-4, upper=1e-1, log=True),
    )

    return search_space


# Designing the run_pipeline
def run_pipeline(**config) -> dict:
    # Create the model
    model = MLP()

    # Epochs to train the model, can be parameterized as fidelity
    max_epochs = 2

    # Load the MNIST dataset for training.
    train_loader, _ = MNIST(batch_size=config["batch_size"])

    # Define the optimizer, criterion, and a scheduler for the learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    # Create the training loop
    for i in range(max_epochs):
        total_loss = 0.0
        incorrect_images = []
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            predicted_labels = torch.argmax(output, dim=1)
            incorrect_mask = predicted_labels != y
            incorrect_images.append(x[incorrect_mask])

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        if len(incorrect_images) > 0:
            incorrect_images = torch.cat(incorrect_images, dim=0)

        ################## TensorBoard Logging Start ##################
        tblogger.log(
            loss=avg_loss,
            current_epoch=i,
            write_summary_incumbent=True,
            writer_config_scalar=True,
            writer_config_hparam=True,
            # Appending extra data
            extra_data={
                "lr_decay": tblogger.scalar_logging(
                    value=scheduler.get_last_lr()[0]
                ),
                "miss_img": tblogger.image_logging(
                    image=incorrect_images, seed=2
                ),
            },
        )
        ################## TensorBoard Logging End ##################
        scheduler.step()
        
        print(f"  Epoch {i + 1} / {max_epochs} Train Error: {avg_loss} ")

    # Retrieve and return the information from the run pipeline.
    return {
        "loss": loss,
        "info_dict": {
            "cost": max_epochs,
        },
    }


# Running the search with random search.
if __name__ == "__main__":
    set_seed(112)
    logging.basicConfig(level=logging.INFO)

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results/mnist_logging_example",
        searcher="random_search",
        max_evaluations_total=5,
    )

```

#### Visualization Results

The following command will open a local host for TensorBoard visualizations, allowing you to view them either in real-time or after the run is complete.

```bash
tensorboard --logdir path/to/root_directory
```

!!! Note
    The annotations within the images provide insights into when and how each graph is triggered.

This image shows visualizations related to scalar values logged during training. Scalars typically include metrics such as loss, incumbent trajectory, a summary of losses for all configurations, and any additional data provided via the `extra_data` argument in the `tblogger.log` function. 

![scalar_loggings](doc_images/tensorboard/tblogger_scalar.jpg)

This image represents visualizations related to logged images during training. It could include snapshots of input data, model predictions, or any other image-related information. In our case, we use images to depict instances of incorrect predictions made by the model.

![image_loggings](doc_images/tensorboard/tblogger_image.jpg)

The following images showcase visualizations related to hyperparameter logging in TensorBoard. These plots include three different views, providing insights into the relationship between different hyperparameters and their impact on the model.

In the table view, you can explore hyperparameter configurations across five different trials. The table displays various hyperparameter values alongside corresponding evaluation metrics.

![hparam_loggings1](doc_images/tensorboard/tblogger_hparam1.jpg)

The parallel coordinate plot offers a holistic perspective on hyperparameter configurations. By presenting multiple hyperparameters simultaneously, this view allows you to observe the interactions between variables, providing insights into their combined influence on the model.

![hparam_loggings2](doc_images/tensorboard/tblogger_hparam2.jpg)

The scatter plot matrix view provides an in-depth analysis of pairwise relationships between different hyperparameters. By visualizing correlations and patterns, this view aids in identifying key interactions that may influence the model's performance.

![hparam_loggings3](doc_images/tensorboard/tblogger_hparam3.jpg)