import logging

import numpy as np
from pathlib import Path
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


# Important: Include the "pipeline_directory" and "previous_pipeline_directory" arguments
# in your evaluate_pipeline function. This grants access to NePS's folder system and is
# critical for leveraging efficient multi-fidelity optimization strategies.
# For more details, refer to the working_directory_per_pipeline example in convenience.


def evaluate_pipeline(
    pipeline_directory: Path,  # The path associated with this configuration
    previous_pipeline_directory: Path
    | None,  # The path associated with any previous config
    learning_rate: float,
    epoch: int,
) -> dict:
    model, optimizer = get_model_and_optimizer(learning_rate)
    checkpoint_name = "checkpoint.pth"

    if previous_pipeline_directory is not None:
        # Read in state of the model after the previous fidelity rung
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs_previously_spent = checkpoint["epoch"]
    else:
        epochs_previously_spent = 0

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

    objective_to_minimize = np.log(learning_rate / epoch)  # Replace with actual error
    epochs_spent_in_this_call = epoch - epochs_previously_spent  # Optional for stopping
    return dict(
        objective_to_minimize=objective_to_minimize, cost=epochs_spent_in_this_call
    )


pipeline_space = dict(
    learning_rate=neps.Float(lower=1e-4, upper=1e0, log=True),
    epoch=neps.Integer(lower=1, upper=10, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multi_fidelity_example",
    # Optional: Do not start another evaluation after <=50 epochs, corresponds to cost
    # field above.
    max_cost_total=50,
)
