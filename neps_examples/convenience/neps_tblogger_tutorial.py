"""
NePS tblogger With TensorBoard
==============================

1- Introduction
---------------
Welcome to the NePS tblogger with TensorBoard tutorial. This guide will walk you
through the process of using the NePS tblogger class to monitor performance
data for different hyperparameter configurations during optimization.

Assuming you have experience with NePS, this tutorial aims to showcase the power
of visualization using tblogger. To go directly to that part, check lines 244-264
or search for 'Start Tensorboard Logging'.

2- Learning Objectives
----------------------
By completing this tutorial, you will:

- Understand the role of NePS tblogger in HPO and NAS.
- Learn to define search spaces within NePS for different model configurations.
- Build a comprehensive run pipeline to train and evaluate models.
- Utilize TensorBoard to visualize and compare performance metrics of different
  model configurations.

3- Setup
--------
Before we begin, ensure you have the necessary dependencies installed. To install
the 'NePS' package, use the following command:

```bash
pip install neural-pipeline-search
```
"""

import logging
import random
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
#5 Design the evaluate pipeline function.
#6 Use neps.run the run the entire search using your specified optimizer.

Each step will be covered in detail throughout the code

"""

#############################################################
# Definig the seeds for reproducibility


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#############################################################
# Prepare the input data. For this tutorial we use the MNIST dataset.


def MNIST(
    batch_size: int = 256,
    n_train_size: float = 0.9,
    data_reduction_factor: float = 0.5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Download MNIST training and test datasets if not already downloaded.
    train_dataset = torchvision.datasets.MNIST(
        root=".data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=".data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Determine the size of the reduced training dataset for faster training
    # and calculate the size of the training subset from the reduced dataset
    reduced_dataset_train = int(data_reduction_factor * len(train_dataset))
    train_size = int(n_train_size * reduced_dataset_train)

    # Create a random sampler for the training and validation data
    train_sampler = SubsetRandomSampler(range(train_size))
    valid_sampler = SubsetRandomSampler(range(train_size, reduced_dataset_train))

    # Create DataLoaders for training, validation, and test datasets.
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


#############################################################
# Design small MLP model to be able to represent the input data.


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


#############################################################
# Define the training step. Return the validation error and
# misclassified images.


def objective_to_minimize_ev(model: nn.Module, data_loader: DataLoader) -> float:
    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for efficiency.
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)

            # Get the predicted class for each input.
            _, predicted = torch.max(output.data, 1)

            # Update the correct and total counts.
            correct += (predicted == y).sum().item()
            total += y.size(0)

    # Calculate the accuracy and return the error rate.
    accuracy = correct / total
    error_rate = 1 - accuracy
    return error_rate


def training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> float:
    """
    Function that trains the model for one epoch and evaluates the model
    on the validation set.

    Args:
        model (nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the weights.
        criterion (nn.Module) : Loss function to use.
        train_loader (DataLoader): DataLoader containing the training data.
        validation_loader (DataLoader): DataLoader containing the validation data.

    Returns:
    float: The validation error (float)
    """
    model.train()

    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        objective_to_minimize = criterion(output, y)
        objective_to_minimize.backward()
        optimizer.step()

    # Calculate validation objective_to_minimize using the objective_to_minimize_ev function.
    validation_objective_to_minimize = objective_to_minimize_ev(
        model, validation_loader
    )
    return validation_objective_to_minimize


#############################################################
# Design the pipeline search spaces.


def pipeline_space() -> dict:
    pipeline = dict(
        lr=neps.Float(lower=1e-5, upper=1e-1, log=True),
        optim=neps.Categorical(choices=["Adam", "SGD"]),
        weight_decay=neps.Float(lower=1e-4, upper=1e-1, log=True),
    )

    return pipeline


#############################################################
# Implement the pipeline run search.
def evaluate_pipeline(lr, optim, weight_decay):
    # Create the network model.
    model = MLP()

    if optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            "Optimizer choices are defined differently in the pipeline_space"
        )

    max_epochs = 2  # Epochs to train the model, can be parameterized as fidelity

    # Load the MNIST dataset for training, validation, and testing.
    train_loader, validation_loader, test_loader = MNIST(
        batch_size=96, n_train_size=0.6, data_reduction_factor=0.75
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)
    criterion = nn.CrossEntropyLoss()

    # Substitute the Tensorboard SummaryWriter with ConfigWriter from NePS
    # writer = SummaryWriter()
    writer = tblogger.ConfigWriter(write_summary_incumbent=True)

    for i in range(max_epochs):
        objective_to_minimize = training(
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )

        # Gathering the gradient mean in each layer
        mean_gradient = []
        for layer in model.children():
            layer_gradients = [param.grad for param in layer.parameters()]
            if layer_gradients:
                mean_gradient.append(
                    torch.mean(torch.cat([grad.view(-1) for grad in layer_gradients]))
                )

        ###################### Start Tensorboard Logging ######################
        # 1. Loss curves of each configuration at each epoch.
        # 2. Decay curve of the learning rate at each epoch.
        # 3. First two layer gradients passed as scalar configs.

        writer.add_scalar(tag="loss", scalar_value=objective_to_minimize, global_step=i)
        writer.add_scalar(
            tag="lr_decay", scalar_value=scheduler.get_last_lr()[0], global_step=i
        )
        writer.add_scalar(
            tag="layer_gradient1", scalar_value=mean_gradient[0], global_step=i
        )
        writer.add_scalar(
            tag="layer_gradient2", scalar_value=mean_gradient[1], global_step=i
        )

        scheduler.step()

        print(f"  Epoch {i + 1} / {max_epochs} Val Error: {objective_to_minimize} ")

    # 4. Hparams comparison.
    writer.add_hparams(
        hparam_dict={"lr": lr, "optim": optim, "wd": weight_decay},
        metric_dict={"loss_val": objective_to_minimize},
    )
    writer.close()

    ###################### End Tensorboard Logging ######################

    train_accuracy = objective_to_minimize_ev(model, train_loader)
    test_accuracy = objective_to_minimize_ev(model, test_loader)

    # Return a dictionary with relevant metrics and information.
    return {
        "objective_to_minimize": objective_to_minimize,
        "info_dict": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cost": max_epochs,
        },
    }


#############################################################
# Running neps with BO as the optimizer.

if __name__ == "__main__":
    """
    When running this code without any arguments, it will by default
    run bayesian optimization with 3 evaluations total.

    ```bash
    python neps_examples\convenience\neps_tblogger_tutorial.py
    ```
    """
    start_time = time.time()

    set_seed(112)
    logging.basicConfig(level=logging.INFO)

    run_args = dict(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results/neps_tblogger_example",
        optimizer="random_search",
    )

    neps.run(
        **run_args,
        evaluations_to_spend=3,
    )

    """
    To check live plots during this search, please open a new terminal
    and make sure to be at the same level directory of your project and
    run the following command on the file created by neps root_directory.
    Running both from root-directory, the command would be:

    ```bash:
    tensorboard --logdir results\neps_tblogger_example
    ```

    To be able to check the visualization of tensorboard make sure to
    follow the local link provided.

    http://localhost:6006/
    """

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    logging.info(f"Execution time: {execution_time} seconds\n")
