from __future__ import annotations

import argparse
from warnings import warn

import numpy as np

import neps


def run_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning, stacklevel=2)
    return evaluate_pipeline_constant(learning_rate, optimizer, epochs, batch_size)

def evaluate_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    """Func for test loading of evaluate_pipeline."""
    eval_score = np.random.choice([learning_rate, epochs], 1) if optimizer == "a" else 5.0
    eval_score += batch_size
    return {"objective_to_minimize": eval_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yml."
    )
    parser.add_argument("run_args", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--evaluate_pipeline", action="store_true")
    args = parser.parse_args()

    if args.evaluate_pipeline:
        neps.run(run_args=args.run_args, evaluate_pipeline=evaluate_pipeline_constant)
    else:
        neps.run(run_args=args.run_args)
