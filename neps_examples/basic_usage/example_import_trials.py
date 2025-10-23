import logging
import numpy as np
import neps
import socket
import os
from neps import UserResultDict
import random
import torch
import argparse
import neps.utils

logging.basicConfig(level=logging.DEBUG)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def get_evaluate_pipeline_func(optimizer):
    match optimizer:
        case "primo":
            def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
                objective_to_minimize = [
                    float1 - 0.3,
                    float2 - 3.6
                ]
                return objective_to_minimize
        case "ifbo":
            def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
                objective_to_minimize = abs(float1) / abs(float(
                    np.sum([float1, float2, int(categorical), integer1, integer2]))
                )
                return objective_to_minimize
        case _:
            def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
                objective_to_minimize = -float(
                    np.sum([float1, float2, int(categorical), integer1, integer2])
                )
                return objective_to_minimize
    return evaluate_pipeline


def get_evaluated_trials(optimizer):
    # Each optimizer gets its own evaluated trials fixture
    match optimizer:
        case "asha":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
            ]
        case "successive_halving":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
            ]
        case "priorband":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
            ]
        case "primo":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=[0.5417078469603, 3.3333333333333335])),
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.6,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=[0.2417078469603, 3.6])),
            ]
        case "ifbo":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=0.5417078469603)),
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.6,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=0.2417078469603)),
            ]
        case "hyperband":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
                ({
                    "float1": 0.5417078469603526,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 800,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
            ]
        case "bayesian_optimization":
            return [
                ({
                    "float1": 0.5884444338738143,
                    "float2": 3.3333333333333335,
                    "categorical": 0,
                    "integer1": 0,
                    "integer2": 1000,
                }, {"objective_to_minimize": -1011.5417078469603}),
            ]
        case "async_hb":
            return [
                ({
                    "float1": 0.5417078469603526,
                    "float2": 3.3333333333333335,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 1000,
                }, UserResultDict(objective_to_minimize=-1011.5417078469603)),
            ]
        
    raise ValueError(f"Unknown optimizer: {optimizer}")

def run_import_trials(optimizer):
    pipeline_space = neps.SearchSpace(
        dict(
            float1=neps.Float(lower=0, upper=1),
            float2=neps.Float(lower=1, upper=10, is_fidelity=True),
            categorical=neps.Categorical(choices=[0, 1]),
            integer1=neps.Integer(lower=0, upper=1),
            integer2=neps.Integer(lower=1, upper=1000, log=True),
        )
    )

    # here we write something 
    neps.run(
        evaluate_pipeline=get_evaluate_pipeline_func(optimizer=optimizer),
        pipeline_space=pipeline_space,
        root_directory=f"initial_results_{optimizer}",
        fidelities_to_spend=5,
        worker_id=f"worker_{optimizer}-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer
    )

    trials = neps.utils.load_trials_from_pickle(root_dir=f"initial_results_{optimizer}")

    # import trials been evaluated above
    neps.import_trials(
        pipeline_space,
        evaluated_trials=trials,
        root_directory=f"results_{optimizer}",
        optimizer=optimizer
    )

    # imort some trials evaluated in some other setup
    neps.import_trials(
        pipeline_space,
        evaluated_trials=get_evaluated_trials(optimizer),
        root_directory=f"results_{optimizer}",
        optimizer=optimizer
    )

    neps.run(
        evaluate_pipeline=get_evaluate_pipeline_func(optimizer=optimizer),
        pipeline_space=pipeline_space,
        root_directory=f"results_{optimizer}",
        fidelities_to_spend=20,
        worker_id=f"worker_{optimizer}_resume-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=[
            "asha", "successive_halving", "priorband", "primo",
            "ifbo", "hyperband", "bayesian_optimization", "async_hb"
        ],
        help="Optimizer to test."
    )
    args = parser.parse_args()
    print(f"Testing import_trials for optimizer: {args.optimizer}")
    run_import_trials(args.optimizer)
