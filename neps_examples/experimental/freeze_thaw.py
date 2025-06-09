import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import neps
from neps import tblogger
from neps.plot.plot3D import Plotter3D


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


def training_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    num_layers,
    num_neurons,
    epochs,
    learning_rate,
    weight_decay,
):
    """
    Trains and validates a simple neural network on the MNIST dataset.

    Args:
        num_layers (int): Number of hidden layers in the network.
        num_neurons (int): Number of neurons in each hidden layer.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Name of the optimizer to use ('adam' or 'sgd').

    Returns:
        float: The average objective_to_minimize over the validation set after training.

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
        root="./.data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    model = SimpleNN(28 * 28, num_layers, num_neurons)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Loading potential checkpoint
    start_epoch = 1
    if previous_pipeline_directory is not None:
        if (Path(previous_pipeline_directory) / "checkpoint.pt").exists():
            states = torch.load(
                Path(previous_pipeline_directory) / "checkpoint.pt",
                weights_only=False
            )
            model = states["model"]
            optimizer = states["optimizer"]
            start_epoch = states["epochs"]

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            objective_to_minimize = criterion(output, target)
            objective_to_minimize.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_objective_to_minimize = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_objective_to_minimize += criterion(output, target).item()

            # Get the predicted class
            _, predicted = torch.max(output.data, 1)

            # Count correct predictions
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

    val_objective_to_minimize /= len(val_loader.dataset)
    val_err = 1 - val_correct / val_total

    # Saving checkpoint
    states = {
        "model": model,
        "optimizer": optimizer,
        "epochs": epochs,
    }
    torch.save(states, Path(pipeline_directory) / "checkpoint.pt")

    # Logging
    # tblogger.log(
    #     objective_to_minimize=val_objective_to_minimize,
    #     current_epoch=epochs,
    #     # Set to `True` for a live incumbent trajectory.
    #     write_summary_incumbent=True,
    #     # Set to `True` for a live objective_to_minimize trajectory for each config.
    #     writer_config_scalar=True,
    #     # Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
    #     writer_config_hparam=True,
    #     # Appending extra data
    #     extra_data={
    #         "train_objective_to_minimize": tblogger.scalar_logging(
    #             objective_to_minimize.item()
    #         ),
    #         "val_err": tblogger.scalar_logging(val_err),
    #     },
    # )

    return val_err


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline_space = {
        "learning_rate": neps.Float(1e-5, 1e-1, log=True),
        "num_layers": neps.Integer(1, 5),
        "num_neurons": neps.Integer(64, 128),
        "weight_decay": neps.Float(1e-5, 0.1, log=True),
        "epochs": neps.Integer(1, 10, is_fidelity=True),
    }

    neps.run(
        pipeline_space=pipeline_space,
        evaluate_pipeline=training_pipeline,
        optimizer="ifbo",
        max_evaluations_total=50,
        root_directory="./debug/ifbo-mnist/",
        overwrite_root_directory=False,  # set to False for a multi-worker run
    )

    # NOTE: this is `experimental` and may not work as expected
    ## plotting a 3D plot for learning curves explored by ifbo
    plotter = Plotter3D(
        run_path="./debug/ifbo-mnist/",  # same as `root_directory` above
        fidelity_key="epochs",  # same as `pipeline_space`
    )
    plotter.plot3D(filename="ifbo")
