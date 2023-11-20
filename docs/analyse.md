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

### 2. Tensorboard

#### Introduction

[TensorBoard](https://www.tensorflow.org/tensorboard) serves as a valuable tool for visualizing machine learning experiments, offering the ability to observe losses and metrics throughout the model training process while also providing visual representations of model architectures. In NePS, we use this powerful tool to show metrics of configurations during training in addition to comparisons to different hyperparameters used in the search for better diagnosis of the model and the search using tensorboard's HParam plugin.

#### `tblogger.log` Function

Logs information to TensorBoard for visualizing the hyperparameter optimization process.

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
    - `writer_config_scalar` (bool, optional): Set to `True` for a live loss trajectory for each configuration.
    - `writer_config_hparam` (bool, optional): Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
    - `extra_data` (dict, optional): Additional data to be logged, provided as a dictionary.

#### `extra_data` Dictionray Content Type

There exist specific functions to be used with the extra_data argument in NePS which are the following:

1- Extra Scalar Logging

- **Signature:**
```python
tblogger.scalar_logging(value: float)
```
- **Parameters:**
    - `value` (float): Any scalar value to be logged at the current epoch of `tblogger.log` function.

2- Extra Image Logging

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
    - `counter` (int): Log image at every multiple of counter.
    - `resize_images` (list of int, optional): List of integers for image sizes after resizing (default: [32, 32]).
    - `random_images` (bool, optional): Images are randomly selected if True (default: True).
    - `num_images` (int, optional): Number of images to log (default: 20).
    - `seed` (int or np.random.RandomState or None, optional): Seed value or RandomState instance to control randomness and reproducibility (default: None).

#### MNIST Example

For illustration purposes, we will employ a straightforward example involving the tuning of hyperparameters for a model utilized in the classification of the MNIST dataset provided by [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

To begin, we'll set up the required imports, establish a seed for reproducibility, and download the necessary training and testing datasets.

```python
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
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    # Download MNIST training and test datasets if not already downloaded.
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Create DataLoaders for training, validation, and test datasets.
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
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

Now, we begin constructing our hyperparameter search algorithm with NePS, guided by three fundamental pillars: the pipeline space, the run_pipeline function, and the neps.run command.

```python
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

    max_epochs = 10  # Epochs to train the model, can be parameterized as fidelity

    # Load the MNIST dataset for training, validation, and testing.
    train_loader, _ = MNIST(batch_size=config["batch_size"])

    # Define the optimizer, criterion, and a scheduler for the learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

    # Create the training loop
    for i in range(max_epochs):
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

        if len(incorrect_images) > 0:
            incorrect_images = torch.cat(incorrect_images, dim=0)
        
        ################## Tensorboard Logging Start ##################
        tblogger.log(
            loss=loss,
            current_epoch=i,
            write_summary_incumbent=True,  # Set to `True` for a live incumbent trajectory.
            writer_config_scalar=True,  # Set to `True` for a live loss trajectory for each config.
            writer_config_hparam=True,  # Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
            # Appending extra data
            extra_data={
                "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0]),
                "miss_img": tblogger.image_logging(image=incorrect_images, counter=2, seed=2),
            },
        )
        ################## Tensorboard Logging End ##################
        scheduler.step()
    
    # Retrieve and return the information from the run pipeline.
    return {
        "loss": loss,
        "info_dict":{
            "cost": max_epochs,
        }    
    }

# Running the search with bayesian optimization.
if __name__ == "__main__":
    set_seed(112)
    logging.basicConfig(level=logging.INFO)

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results/mnist_logging_example",
        searcher="bayesian_optimization",
        max_evaluations_total=20,
    )
```

#### Visualization Results