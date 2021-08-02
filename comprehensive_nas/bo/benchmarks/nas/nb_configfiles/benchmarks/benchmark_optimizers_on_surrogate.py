import json
import os

from functools import partial

import click

from nas_benchmark.discrete_optimizers.run_bohb import bohb_frontend
from nas_benchmark.discrete_optimizers.run_regularized_evolution import (
    random_search_frontend,
)
from nas_benchmark.discrete_optimizers.run_regularized_evolution import (
    regularized_evolution_frontend,
)


optimizers = {
    "Regularized Evolution": regularized_evolution_frontend,
    "Random Search": random_search_frontend,
    # 'SMAC': smac_frontend
    "BOHB (NAS)": partial(bohb_frontend, benchmark="NASBench301_fixed_hps"),
    "BOHB (NAS + HP)": partial(bohb_frontend, benchmark="NASBench301"),
}


@click.command()
@click.option(
    "--optimizer_name",
    type=click.Choice(list(optimizers.keys())),
    help="Which optimizer to use",
    required=True,
)
@click.option("--surrogate_model_dir", type=click.STRING, help="Path to surrogate model.")
@click.option("--runtime_model_dir", type=click.STRING, help="Path to runtime model.")
@click.option(
    "--n_iters",
    type=click.INT,
    help="How many iterations should be performed.",
    default=3000,
)
@click.option(
    "--n_repetitions",
    type=click.INT,
    help="Number of repetitions to be performed.",
    default=10,
)
def benchmark_optimizers_on_surrogate(
    optimizer_name, surrogate_model_dir, runtime_model_dir, n_iters, n_repetitions
):
    log_dir = os.path.join(
        surrogate_model_dir,
        "nas_benchmarks",
        "n_iters_{}_n_repetitions_{}".format(n_iters, n_repetitions),
    )
    # Save the results
    os.makedirs(log_dir, exist_ok=True)
    # Run the surrogate model for every optimizer
    # Potentially parallelize this.
    optimizer = optimizers[optimizer_name]
    print("OPTIMIZER", optimizer_name)
    results = {
        "surrogate_model_dir": surrogate_model_dir,
        "runtime_model_dir": runtime_model_dir,
        "algo": optimizer_name,
        "results": optimizer(
            surrogate_model_dir=surrogate_model_dir,
            runtime_model_dir=runtime_model_dir,
            n_iters=n_iters,
            n_repetitions=n_repetitions,
        ),
    }

    # Save the results
    json.dump(
        results,
        open(
            os.path.join(log_dir, "optimizer_results_{}.json".format(optimizer_name)), "w"
        ),
    )


if __name__ == "__main__":
    benchmark_optimizers_on_surrogate()
