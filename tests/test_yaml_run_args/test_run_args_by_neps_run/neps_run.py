import neps
import numpy as np
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.search_spaces.search_space import SearchSpace
import argparse


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
    learning_rate=neps.FloatParameter(lower=1e-6, upper=1e-1, log=False),
    epochs=neps.IntegerParameter(lower=1, upper=3, is_fidelity=False),
    optimizer=neps.CategoricalParameter(choices=["a", "b", "c"]),
    batch_size=neps.ConstantParameter(64)
)

# Required for testing yaml loading, in the case 'searcher' is an instance of
# BaseOptimizer.
search_space = SearchSpace(**pipeline_space)
optimizer = BayesianOptimization(search_space)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yml.")
    parser.add_argument('run_args', type=str,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()
    neps.run(run_args=args.run_args)
