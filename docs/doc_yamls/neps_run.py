import argparse
import numpy as np
import neps


def run_pipeline(learning_rate, optimizer, epochs):
    """func for test loading of run_pipeline"""
    if optimizer == "a":
        eval_score = np.random.choice([learning_rate, epochs], 1)
    else:
        eval_score = 5.0
    return {"loss": eval_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yml."
    )
    parser.add_argument("run_args", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    neps.run(run_args=args.run_args)
