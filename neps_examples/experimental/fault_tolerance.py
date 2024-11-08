""" To test the fault tolerance, run this script multiple times.
"""

import logging
from warnings import warn

import torch
import torch.nn.functional as F
from torch import nn, optim

import neps


class TheModelClass(nn.Module):
    """Taken from https://pytorch.org/tutorials/beginner/saving_loading_models.html"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_and_optimizer(learning_rate):
    """Taken from https://pytorch.org/tutorials/beginner/saving_loading_models.html"""
    model = TheModelClass()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def run_pipeline(pipeline_directory, learning_rate):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(pipeline_directory, learning_rate)

def evaluate_pipeline(pipeline_directory, learning_rate):
    model, optimizer = get_model_and_optimizer(learning_rate)
    checkpoint_path = pipeline_directory / "checkpoint.pth"

    # Check if there is a previous state of the model training that crashed
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_already_trained = checkpoint["epoch"]
        print(f"Read in model trained for {epoch_already_trained} epochs")
    else:
        epoch_already_trained = 0

    for epoch in range(epoch_already_trained, 101):
        epoch += 1

        # Train model here ....

        # Repeatedly save your progress
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

        # Here we simulate a crash! E.g., due to job runtime limits
        if epoch == 50 and learning_rate < 0.2:
            print("Oh no! A simulated crash!")
            exit()

    return learning_rate  # Replace with actual error


pipeline_space = dict(
    learning_rate=neps.Float(lower=0, upper=1),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/fault_tolerance_example",
    max_evaluations_total=15,
)
previous_results, pending_configs = neps.status("results/fault_tolerance_example")
