import argparse
import neps
import numpy as np


def run_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    """func for test loading of run_pipeline"""
    if optimizer == "a":
        eval_score = np.random.choice([learning_rate, epochs], 1)
    else:
        eval_score = 5.0
    eval_score += batch_size
    return {"loss": eval_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yml."
    )
    parser.add_argument("run_args", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--run_pipeline", action="store_true")
    args = parser.parse_args()

    if args.run_pipeline:
        neps.run(run_args=args.run_args, run_pipeline=run_pipeline_constant)
    else:
        neps.run(run_args=args.run_args)
