""" To test the fault tolerance, run this script multiple times.
"""

import logging

import torch

import neps
from neps_examples.experimental.fault_tolerance_model_and_optimizer import (
    get_model_and_optimizer,
)


def run_pipeline(pipeline_directory, learning_rate):
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
    learning_rate=neps.FloatParameter(lower=0, upper=1),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/fault_tolerance_example",
    max_evaluations_total=15,
)
previous_results, pending_configs = neps.status("results/fault_tolerance_example")
