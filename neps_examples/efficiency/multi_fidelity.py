import logging

import numpy as np
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


def run_pipeline(pipeline_directory, previous_pipeline_directory, learning_rate, epoch):
    model, optimizer = get_model_and_optimizer(learning_rate)
    checkpoint_name = "checkpoint.pth"

    # Read in state of the model after the previous fidelity rung
    if previous_pipeline_directory is not None:
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_already_trained = checkpoint["epoch"]
        print(f"Read in model trained for {epoch_already_trained} epochs")

    # Train model here ...

    # Save model to disk
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pipeline_directory / checkpoint_name,
    )

    return np.log(learning_rate / epoch)  # Replace with actual error


pipeline_space = dict(
    learning_rate=neps.FloatParameter(
        lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
    ),
    epoch=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multi_fidelity",
    max_evaluations_total=20,
)
