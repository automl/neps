"""
NePS tblogger With TensorBoard
====================================
This tutorial demonstrates how to use TensorBoard plugin with NePS tblogger class
to detect performance data of the different model configurations during training.


Setup
-----
To install ``torchvision`` and ``tensorboard`` use the following command:

.. code-block::

   pip install torchvision

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
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

import neps
from neps.plot.tensorboard_eval import tblogger

"""
Steps:

#1 Define the seeds for reproducibility.
#2 Prepare the input data.
#3 Design the model.
#4 Design the pipeline search spaces.
#5 Design the run pipeline function.
#6 Use neps.run the run the entire search using your specified searcher.

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
# Design small MLP model to be able to represent the input data.


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=784, out_features=392)
        self.linear2 = nn.Linear(in_features=392, out_features=196)
        self.linear3 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


#############################################################
# Define the training step and return the validation error and misclassified images.


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
    Function that trains the model for one epoch and evaluates the model on the validation set. Used by the searcher.

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
# Design the pipeline search spaces.


# For BO:
def pipeline_space_BO() -> dict:
    pipeline = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
        optim=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        weight_decay=neps.FloatParameter(lower=1e-4, upper=1e-1, log=True),
    )

    return pipeline


# For Hyperband
def pipeline_space_Hyperband() -> dict:
    pipeline = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
        optim=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        weight_decay=neps.FloatParameter(lower=1e-4, upper=1e-1, log=True),
        epochs=neps.IntegerParameter(lower=1, upper=9, is_fidelity=True),
    )

    return pipeline


#############################################################
# Implement the pipeline run search.


# For BO:
def run_pipeline_BO(lr, optim, weight_decay):
    model = MLP()

    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = 9

    train_loader, validation_loader, test_loader = MNIST(
        batch_size=64, n_train=4096, n_valid=512
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

    criterion = nn.CrossEntropyLoss()
    losses = []

    tblogger.disable(False)

    for i in range(max_epochs):
        loss, miss_img = training(
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        losses.append(loss)

        tblogger.log(
            loss=loss,
            current_epoch=i,
            data={
                "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0]),
                "miss_img": tblogger.image_logging(img_tensor=miss_img, counter=2),
                "layer_gradient": tblogger.layer_gradient_logging(model=model),
            },
        )

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


# For Hyperband
def run_pipeline_Hyperband(pipeline_directory, previous_pipeline_directory, **configs):
    model = MLP()
    checkpoint_name = "checkpoint.pth"
    start_epoch = 0

    train_loader, validation_loader, test_loader = MNIST(
        batch_size=32, n_train=4096, n_valid=512
    )

    # define loss
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    if configs["optim"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
        )
    elif configs["optim"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
        )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)

    # We make use of checkpointing to resume training models on higher fidelities
    if previous_pipeline_directory is not None:
        # Read in state of the model after the previous fidelity rung
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs_previously_spent = checkpoint["epoch"]
    else:
        epochs_previously_spent = 0

    start_epoch += epochs_previously_spent

    losses = list()

    tblogger.disable(False)

    epochs = configs["epochs"]

    for epoch in range(start_epoch, epochs):
        # Call the training function, get the validation errors and append them to val errors
        loss, miss_img = training(
            model, optimizer, criterion, train_loader, validation_loader
        )
        losses.append(loss)

        tblogger.log(
            loss=loss,
            current_epoch=epoch,
            hparam_accuracy_mode=True,
            data={
                "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0]),
                "miss_img": tblogger.image_logging(img_tensor=miss_img, counter=2),
                "layer_gradient": tblogger.layer_gradient_logging(model=model),
            },
        )

        scheduler.step()

        print(f"  Epoch {epoch + 1} / {epochs} Val Error: {loss} ")

    train_accuracy = loss_ev(model, train_loader)
    test_accuracy = loss_ev(model, test_loader)

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pipeline_directory / checkpoint_name,
    )

    return {
        "loss": loss,
        "info_dict": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "val_errors": losses,
            "cost": epochs - epochs_previously_spent,
        },
        "cost": epochs - epochs_previously_spent,
    }


#############################################################
"""
Defining the main with argument parsing to use either BO or Hyperband and specifying their
respective properties
"""

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--searcher",
        type=str,
        choices=["bayesian_optimization", "hyperband"],
        default="bayesian_optimization",
        help="Searcher type used",
    )
    argParser.add_argument(
        "--max_cost_total", type=int, default=30, help="Max cost used for Hyperband"
    )
    argParser.add_argument(
        "--max_evaluations_total", type=int, default=10, help="Max evaluation used for BO"
    )
    args = argParser.parse_args()

    if args.searcher == "hyperband":
        start_time = time.time()
        set_seed(112)
        logging.basicConfig(level=logging.INFO)
        if os.path.exists("results/hyperband"):
            shutil.rmtree("results/hyperband")
        neps.run(
            run_pipeline=run_pipeline_Hyperband,
            pipeline_space=pipeline_space_Hyperband(),
            root_directory="hyperband",
            max_cost_total=args.max_cost_total,
            searcher="hyperband",
        )

        """
        To check live plots during this command run, please open a new terminal with the directory of this saved project and run

                    tensorboard --logdir hyperband
        """

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    elif args.searcher == "bayesian_optimization":
        start_time = time.time()
        set_seed(112)
        logging.basicConfig(level=logging.INFO)
        if os.path.exists("results/bayesian_optimization"):
            shutil.rmtree("results/bayesian_optimization")
        neps.run(
            run_pipeline=run_pipeline_BO,
            pipeline_space=pipeline_space_BO(),
            root_directory="bayesian_optimization",
            max_evaluations_total=args.max_evaluations_total,
            searcher="bayesian_optimization",
        )

        """
        To check live plots during this command run, please open a new terminal with the directory of this saved project and run

                    tensorboard --logdir bayesian_optimization
        """

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    """
    When running this code without any arguments, it will by default run bayesian optimization with 10 max evaluations
    of 9 epochs each:

                python neps_tblogger_tutorial.py


    If you wish to do this run with hyperband searcher with default max cost total of 30. Please run this command on the terminal:

                python neps_tblogger_tutorial.py --searcher hyperband
    """
