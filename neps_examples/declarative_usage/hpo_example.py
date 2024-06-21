import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import neps

"""
This script demonstrates the integration of a simple neural network training pipeline
with NePS for hyperparameter optimization, focusing on the MNIST dataset.

- SimpleNN Class: A PyTorch neural network model that constructs a feedforward
  architecture based on input size, number of layers, and neurons per layer.

- Training Pipeline: A function that takes hyperparameters (number of layers, neurons,
  epochs, learning rate, optimizer type) to train and validate the SimpleNN model on
  the MNIST dataset. Supports Adam and SGD optimizers.

- NEPS Integration: Shows how to automate hyperparameter tuning using NEPS. Configuration
  settings are specified in a YAML file ('run_args.yaml'), which is passed to the NePS
  optimization process via the `run_args` parameter.

Usage:
1. Define model architecture and training logic in `SimpleNN` and `training_pipeline`.
2. Configure hyperparameters and optimization settings in 'config.yaml'.
3. Launch optimization with NePS by calling `neps.run`, specifying the training pipeline,
and configuration file(config.yaml).
"""


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super().__init__()
        layers = [nn.Flatten()]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.ReLU())
            input_size = num_neurons  # Set input size for the next layer

        layers.append(nn.Linear(num_neurons, 10))  # Output layer for 10 classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def training_pipeline(num_layers, num_neurons, epochs, learning_rate, optimizer):
    """
    Trains and validates a simple neural network on the MNIST dataset.

    Args:
        num_layers (int): Number of hidden layers in the network.
        num_neurons (int): Number of neurons in each hidden layer.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Name of the optimizer to use ('adam' or 'sgd').

    Returns:
        float: The average loss over the validation set after training.

    Raises:
        KeyError: If the specified optimizer is not supported.
    """
    # Transformations applied on each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Mean and Std Deviation for MNIST
        ]
    )

    # Loading MNIST dataset
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    model = SimpleNN(28 * 28, num_layers, num_neurons)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise KeyError(f"optimizer {optimizer} is not available")

    # Training loop

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader.dataset)
    return val_loss


if __name__ == "__main__":
    # Configure logging to display important messages from NePS.
    logging.basicConfig(level=logging.INFO)

    # Run optimization using neps.run(...). Arguments can be provided directly to neps.run
    # or defined in a configuration file (e.g., "config.yaml") passed through
    # the run_args parameter.
    neps.run(training_pipeline, run_args="config.yaml")
