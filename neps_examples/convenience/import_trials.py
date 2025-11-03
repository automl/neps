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
from typing import Any

logging.basicConfig(level=logging.INFO)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def get_evaluate_pipeline_func(optimizer):
    match optimizer:
        case "primo":

            def evaluate_pipeline_MO(float1, float2, **kwargs):
                objective_to_minimize = [float1 - 0.3, float2 - 3.6]
                return objective_to_minimize

            evaluate_pipeline = evaluate_pipeline_MO

        case "ifbo":

            def evaluate_pipeline_IFBO(float1, float2, categorical, integer1, integer2):
                objective_to_minimize = abs(float1) / abs(
                    float(np.sum([float1, float2, int(categorical), integer1, integer2]))
                )
                return objective_to_minimize

            evaluate_pipeline = evaluate_pipeline_IFBO

        case _:

            def evaluate_pipeline_default(
                float1, float2, categorical, integer1, integer2
            ):
                objective_to_minimize = -float(
                    np.sum([float1, float2, int(categorical), integer1, integer2])
                )
                return objective_to_minimize

            evaluate_pipeline = evaluate_pipeline_default

    return evaluate_pipeline


def get_evaluated_trials(optimizer) -> list[tuple[dict[str, Any], UserResultDict]]:
    # Common config used by multiple optimizers
    classic_base_config = {
        "float1": 0.5417078469603526,
        "float2": 3.3333333333333335,
        "categorical": 1,
        "integer1": 0,
        "integer2": 1000,
    }
    neps_base_config = {
        "ENVIRONMENT__float2": 1,
        "SAMPLING__Resolvable.categorical::categorical__2": 0,
        "SAMPLING__Resolvable.float1::float__0_1_False": 0.5,
        "SAMPLING__Resolvable.integer1::integer__0_1_False": 1,
        "SAMPLING__Resolvable.integer2::integer__1_1000_True": 5,
    }
    base_result = UserResultDict(objective_to_minimize=-1011.5417078469603)

    # Mapping of optimizers to their evaluated trials
    trials_map = {
        "asha": [(classic_base_config, base_result)],
        "successive_halving": [(classic_base_config, base_result)],
        "priorband": [(classic_base_config, base_result)],
        "primo": [
            (
                classic_base_config,
                UserResultDict(
                    objective_to_minimize=[0.5417078469603, 3.3333333333333335]
                ),
            ),
            (
                {**classic_base_config, "float2": 3.6},
                UserResultDict(objective_to_minimize=[0.2417078469603, 3.6]),
            ),
        ],
        "ifbo": [
            (classic_base_config, UserResultDict(objective_to_minimize=0.5417078469603)),
            (
                {**classic_base_config, "float2": 3.6},
                UserResultDict(objective_to_minimize=0.2417078469603),
            ),
        ],
        "hyperband": [
            (classic_base_config, base_result),
            (
                {
                    "float1": 0.5417078469603526,
                    "categorical": 1,
                    "integer1": 0,
                    "integer2": 800,
                },
                base_result,
            ),
        ],
        "bayesian_optimization": [
            (
                {
                    "float1": 0.5884444338738143,
                    "float2": 3.3333333333333335,
                    "categorical": 0,
                    "integer1": 0,
                    "integer2": 1000,
                },
                {"objective_to_minimize": -1011.5417078469603},
            ),
        ],
        "async_hb": [(classic_base_config, base_result)],
        "neps_hyperband": [(neps_base_config, base_result)],
        "neps_priorband": [(neps_base_config, base_result)],
    }

    if optimizer not in trials_map:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return trials_map[optimizer]


def run_import_trials(optimizer):
    class ExampleSpace(neps.PipelineSpace):
        float1 = neps.Float(lower=0, upper=1)
        float2 = neps.Fidelity(neps.Float(lower=1, upper=10))
        categorical = neps.Categorical(choices=[0, 1])
        integer1 = neps.Integer(lower=0, upper=1)
        integer2 = neps.Integer(lower=1, upper=1000, log=True)


    logging.info(
        f"{'-'*80} Running initial evaluations for optimizer {optimizer}. {'-'*80}"
    )

    # here we write something
    neps.run(
        evaluate_pipeline=get_evaluate_pipeline_func(optimizer=optimizer),
        pipeline_space=ExampleSpace(),
        root_directory=f"results/trial_import/initial_results_{optimizer}",
        overwrite_root_directory=True,
        fidelities_to_spend=5,
        worker_id=f"worker_{optimizer}-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer,
    )

    trials = neps.utils.load_trials_from_pickle(
        root_dir=f"results/trial_import/initial_results_{optimizer}"
    )

    logging.info(
        f"{'-'*80} Importing {len(trials)} trials for optimizer {optimizer}. {'-'*80}"
    )

    # import trials been evaluated above
    neps.import_trials(
        ExampleSpace(),
        evaluated_trials=trials,
        root_directory=f"results/trial_import/results_{optimizer}",
        overwrite_root_directory=True,
        optimizer=optimizer,
    )

    logging.info(
        f"{'-'*80} Importing {len(get_evaluated_trials(optimizer))} trials for optimizer"
        f" {optimizer}. {'-'*80}"
    )

    # import some trials evaluated in some other setup
    neps.import_trials(
        ExampleSpace(),
        evaluated_trials=get_evaluated_trials(optimizer),
        root_directory=f"results/trial_import/results_{optimizer}",
        optimizer=optimizer,
    )

    logging.info(f"{'-'*80} Running after import for optimizer {optimizer}. {'-'*80}")

    neps.run(
        evaluate_pipeline=get_evaluate_pipeline_func(optimizer=optimizer),
        pipeline_space=ExampleSpace(),
        root_directory=f"results/trial_import/results_{optimizer}",
        fidelities_to_spend=10,
        worker_id=f"worker_{optimizer}_resume-{socket.gethostname()}-{os.getpid()}",
        optimizer=optimizer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=[
            "asha",
            "successive_halving",
            "priorband",
            "primo",
            "ifbo",
            "hyperband",
            "bayesian_optimization",
            "async_hb",
            "neps_hyperband",
            "neps_priorband",
        ],
        help="Optimizer to test.",
    )
    args = parser.parse_args()
    print(f"Testing import_trials for optimizer: {args.optimizer}")
    run_import_trials(args.optimizer)
