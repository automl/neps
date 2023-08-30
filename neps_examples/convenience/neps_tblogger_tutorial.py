"""
NePS tblogger With TensorBoard
====================================
1- Introduction
---------------

Welcome to the NePS tblogger with TensorBoard tutorial. This guide will walk you
through the process of using the NePS tblogger class to effectively monitor and
analyze performance data from various model configurations during training.

Assuming you already have experience in NePS, the main reason of creating this tutorial is to showcase the
power of visualization using tblogger. if you wish to directly reach that part, check the lines
between 244-264 or search for 'Start Tensorboard Logging'

2- Learning Objectives
----------------------

By completing this tutorial, you will:

- Understand the role of NePS tblogger and its importance in HPO and NAS.
- Learn how to define search spaces within NePS to explore different model configurations.
- Build a comprehensive run pipeline to train and evaluate models.
- Utilize TensorBoard to visualize and compare performance metrics of different model configurations.

3- Setup
--------

Before we dive in, make sure you have the necessary dependencies installed. If you haven't already,
install the ``NePS`` package using the following command:

```bash
pip install neural-pipeline-search
```

Additionally, please note that NePS does not include ``torchvision`` as a dependency.
You can install it with the following command:

```bash
pip install torchvision==0.14.1
```

These dependencies will ensure you have everything you need to follow along with this tutorial successfully.
"""

import argparse
import logging
import os
import random
import shutil
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

import neps
from neps.plot.tensorboard_eval import tblogger

"""
Steps for a successful training pipeline:

#1 Define the seeds for reproducibility.
#2 Prepare the input data.
#3 Design the model.
#4 Design the pipeline search spaces.
#5 Design the run pipeline function.
#6 Use neps.run the run the entire search using your specified searcher.

Each step will be covered in detail thourghout the code

"""

#############################################################
# 1 Definig the seeds for reproducibility


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#############################################################
# 2 Prepare the input data. For this tutorial we use the MNIST dataset.


def MNIST(
    batch_size: int = 32, n_train: int = 8192, n_valid: int = 1024
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    train_sampler = SubsetRandomSampler(range(n_train))
    valid_sampler = SubsetRandomSampler(range(n_train, n_train + n_valid))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler
    )
    val_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


#############################################################
# 3 Design small MLP model to be able to represent the input data.


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=392)
        self.linear2 = nn.Linear(in_features=392, out_features=196)
        self.linear3 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x):
        # Flattening the grayscaled image from 1x28x28 (CxWxH) to 784.
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


#############################################################
# 4 Define the training step and return the validation error and misclassified images.


def loss_ev(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return 1 - accuracy


def training(model, optimizer, criterion, train_loader, validation_loader):
    """
    Function that trains the model for one epoch and evaluates the model on the validation set.

    Args:
        model (nn.Module): Model to be trained.
        optimizer (torch.nn.optim): Optimizer used to train the weights (depends on the pipeline space).
        criterion (nn.modules.loss) : Loss function to use.
        train_loader (torch.utils.Dataloader): Data loader containing the training data.
        validation_loader (torch.utils.Dataloader): Data loader containing the validation data.

    Returns:
        (float) validation error for the epoch.
    """
    incorrect_images = []
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        predicted_labels = torch.argmax(output, dim=1)
        incorrect_mask = predicted_labels != y
        incorrect_images.append(x[incorrect_mask])

    validation_loss = loss_ev(model, validation_loader)

    if len(incorrect_images) > 0:
        incorrect_images = torch.cat(incorrect_images, dim=0)

    return validation_loss, incorrect_images


#############################################################
# 5 Design the pipeline search spaces.


def pipeline_space_BO() -> dict:
    pipeline = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
        optim=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        weight_decay=neps.FloatParameter(lower=1e-4, upper=1e-1, log=True),
    )

    return pipeline


#############################################################
# 6 Implement the pipeline run search.


def run_pipeline_BO(lr, optim, weight_decay):
    model = MLP()

    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            "Optimizer choices are defined differently in the pipeline_space"
        )

    max_epochs = 9

    train_loader, validation_loader, test_loader = MNIST(
        batch_size=64, n_train=4096, n_valid=512
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for i in range(max_epochs):
        loss, miss_img = training(
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        losses.append(loss)

        # Gathering the gradient mean in each layer to display some of them in tensorboard
        mean_gradient = []
        for layer in model.children():
            layer_gradients = [param.grad for param in layer.parameters()]
            if layer_gradients:
                mean_gradient.append(
                    torch.mean(torch.cat([grad.view(-1) for grad in layer_gradients]))
                )

        ###################### Start Tensorboard Logging ######################

        # tblogger for neps config loggings. This line will result in the following:

        # 1 Incumbent of the configs (best performance regardless of fidelity budget, if the searcher was fidelity dependent).
        # 2 Loss curves of each of the configs at each epoch.
        # 3 lr_decay curve at each epoch.
        # 4 miss_img which represents the wrongly classified images by the model.
        # 5 first two layer_gradients computed above and passed as scalar configs.

        tblogger.log(
            loss=loss,
            current_epoch=i,
            data={
                "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0]),
                "miss_img": tblogger.image_logging(
                    img_tensor=miss_img, counter=2, seed=2
                ),
                "layer_gradient1": tblogger.scalar_logging(value=mean_gradient[0]),
                "layer_gradient2": tblogger.scalar_logging(value=mean_gradient[1]),
            },
        )

        ###################### End Tensorboard Logging ######################

        scheduler.step()

        print(f"  Epoch {i + 1} / {max_epochs} Val Error: {loss} ")

    train_accuracy = loss_ev(model, train_loader)
    test_accuracy = loss_ev(model, test_loader)

    return {
        "loss": loss,
        "info_dict": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "val_errors": losses,
            "cost": max_epochs,
        },
    }


#############################################################
# 6 Running neps with BO as our main searcher, saving the results in a defined directory.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_evaluations_total",
        type=int,
        default=10,
        help="Number of different configs to train",
    )
    args = parser.parse_args()

    """
    When running this code without any arguments, it will by default run bayesian optimization with 10 max evaluations
    of 9 epochs each:

    ```bash
    python neps_tblogger_tutorial.py
    ```
    """

    start_time = time.time()

    set_seed(112)
    logging.basicConfig(level=logging.INFO)

    if os.path.exists("results/bayesian_optimization"):
        shutil.rmtree("results/bayesian_optimization")

    """
    For showcasing purposes. After completing the first run, one can uncomment line 351
    and continue the search via:

    ```bash:
    python neps_tblogger_tutorial.py --max_evaluations_total 15
    ```

    This would result in continuing the search for 5 new different configurations in addition
    to disabling the logging, hence tblogger can always be disabled using the line below.

    ```python:
    tblogger.disable()
    ```

    Note that by default tblogger is enabled when used. However, there is also an enable toggle that can be used
    ```python:
    tblogger.enable()
    ```
    """

    # by defualt tblogger is enabled when used, one can also check the status using:
    # tblogger.get_status()

    # tblogger.disable()

    neps.run(
        run_pipeline=run_pipeline_BO,
        pipeline_space=pipeline_space_BO(),
        root_directory="bayesian_optimization",
        max_evaluations_total=args.max_evaluations_total,
        searcher="bayesian_optimization",
    )

    """
    To check live plots during this search, please open a new terminal and make sure to be at the same level directory
    of your project and run this command on the file created by neps root_directory.

    ```bash:
    tensorboard --logdir bayesian_optimization
    ```

    To be able to check the visualization of tensorboard make sure to follow the local link provided.

    ```bash:
    http://localhost:6006/
    ```

    If nothing was visualized and you followed the tutorial exactly, there could have been an error in passing the correct
    directory, please double check. Tensorboard will always run in the command line without checking if the directory exists.
    """

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    logging.info(f"Execution time: {execution_time} seconds")