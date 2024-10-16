import argparse
import numpy as np
import neps


def run_pipeline(learning_rate, epochs, optimizer, batch_size):
    """func for test loading of run_pipeline"""
    if optimizer == "a":
        eval_score = np.random.choice([learning_rate, epochs], 1)
    else:
        eval_score = 5.0
    eval_score += batch_size
    return {"loss": eval_score}


# For testing the functionality of loading a dictionary from a YAML configuration.
pipeline_space = dict(
    learning_rate=neps.Float(lower=1e-6, upper=1e-1, log=False),
    epochs=neps.Integer(lower=1, upper=3, is_fidelity=False),
    optimizer=neps.Categorical(choices=["a", "b", "c"]),
    batch_size=neps.Constant(64),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yaml.")
    parser.add_argument("run_args", type=str,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--kwargs_flag", action="store_true",
                        help="flag for adding kwargs")
    args = parser.parse_args()

    hyperband_args_optimizer = {"random_interleave_prob": 0.9,
                                "sample_default_first": False,
                                "sample_default_at_target": False,
                                "eta": 7}

    if args.kwargs_flag:
        neps.run(run_args=args.run_args, **hyperband_args_optimizer)
    else:
        neps.run(run_args=args.run_args)
