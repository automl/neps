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

    loss = np.log(learning_rate / epoch)  # Replace with actual error
    epochs_spent_in_this_call = epoch - epochs_previously_spent  # Optional for stopping
    learning_curve = np.linspace(
        0, loss, num=int(epoch)
    ).tolist()  # learning curves as a list
    return dict(loss=loss, cost=epochs_spent_in_this_call, learning_curve=learning_curve)


pipeline_space = dict(
    learning_rate=neps.FloatParameter(lower=1e-4, upper=1e0, log=True),
    epoch=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multi_fidelity_example",
    searcher="mf_ei_bo",
    # Optional: Do not start another evaluation after <=100 epochs, corresponds to cost
    # field above.
    max_cost_total=50,
    surrogate_model="deep_gp",
    # Normalizing y here since we return unbounded loss, not completely correct to do so
    surrogate_model_args={
        "surrogate_model_fit_args": {"normalize_y": True},
    },
)
