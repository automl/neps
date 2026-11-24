import logging
import numpy as np
import scalefm
import socket
import os
from scalefm import UserResultDict
import random
import torch
import argparse
import scalefm.utils
from typing import Any

logging.basicConfig(level=logging.INFO)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    return {"objective_to_minimize": objective_to_minimize, "cost": 1.0}


def get_evaluated_trials() -> list[tuple[dict[str, Any], UserResultDict]]:
    """Trials evaluated in some other setup."""
    return [
        (
            {
                "float1": 0.5417078469603526,
                "float2": 3.3333333333333335,
                "categorical": 1,
                "integer1": 0,
                "integer2": 1000,
            },
            UserResultDict(objective_to_minimize=-1005.8750411802937, cost=1.0),
        ),
        (
            {
                "float1": 0.5884444338738143,
                "float2": 3.6,
                "categorical": 0,
                "integer1": 0,
                "integer2": 800,
            },
            UserResultDict(objective_to_minimize=-804.1884444338738, cost=1.0),
        ),
    ]


def run_import_trials(optimizer):
    class ExampleSpace(scalefm.PipelineSpace):
        float1 = scalefm.Float(lower=0, upper=1)
        float2 = scalefm.Float(lower=1, upper=10)
        categorical = scalefm.Categorical(choices=[0, 1])
        integer1 = scalefm.Integer(lower=0, upper=1)
        integer2 = scalefm.Integer(lower=1, upper=1000, log=True)

    logging.info(
        f"{'-' * 80} Running initial evaluations for optimizer {optimizer}. {'-' * 80}"
    )

    # here we write something
    scalefm.acquire(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=ExampleSpace(),
        root_directory=f"results/trial_import/initial_results_{optimizer}",
        overwrite_root_directory=True,
        worker_evaluations_to_spend=5,
        worker_id=f"worker_{optimizer}-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer,
    )

    trials = scalefm.utils.load_trials_from_pickle(
        root_dir=f"results/trial_import/initial_results_{optimizer}"
    )

    logging.info(
        f"{'-' * 80} Importing {len(trials)} trials for optimizer {optimizer}. {'-' * 80}"
    )

    # import trials been evaluated above
    scalefm.import_trials(
        evaluated_trials=trials,
        root_directory=f"results/trial_import/results_{optimizer}",
        pipeline_space=ExampleSpace(),
        overwrite_root_directory=True,
        optimizer=optimizer,
    )

    logging.info(
        f"{'-' * 80} Importing {len(get_evaluated_trials())} trials for optimizer"
        f" {optimizer}. {'-' * 80}"
    )

    # import some trials evaluated in some other setup
    scalefm.import_trials(
        evaluated_trials=get_evaluated_trials(),
        root_directory=f"results/trial_import/results_{optimizer}",
        pipeline_space=ExampleSpace(),
        optimizer=optimizer,
    )

    logging.info(f"{'-' * 80} Running after import for optimizer {optimizer}. {'-' * 80}")

    scalefm.acquire(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=ExampleSpace(),
        root_directory=f"results/trial_import/results_{optimizer}",
        worker_evaluations_to_spend=10,
        worker_id=f"worker_{optimizer}_resume-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["random_search", "bayesian_optimization"],
        help="Optimizer to test.",
    )
    args = parser.parse_args()
    print(f"Testing import_trials for optimizer: {args.optimizer}")
    run_import_trials(args.optimizer)
