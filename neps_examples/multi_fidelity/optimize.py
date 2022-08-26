import argparse
import logging

import numpy as np
import torch

import neps
from neps_examples.multi_fidelity.model_and_optimizer import get_model_and_optimizer

multi_fidelity_algos = [
    "successive_halving",
    "successive_halving_prior",
    "asha",
    "asha_prior",
]


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


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", type=str, default="successive_halving", choices=multi_fidelity_algos
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_args()

    pipeline_space = dict(
        learning_rate=neps.FloatParameter(
            lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
        ),
        epoch=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
    )

    logging.basicConfig(level=logging.INFO)

    searcher = args.algo
    searcher_kwargs = dict(use_priors=False) if "prior" not in args.algo else dict()
    searcher_output = f"{args.algo}_example"
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"results/{searcher_output}",
        max_evaluations_total=50,
        searcher=searcher,
        **searcher_kwargs,
    )
    previous_results, pending_configs = neps.status(f"results/{searcher_output}")
