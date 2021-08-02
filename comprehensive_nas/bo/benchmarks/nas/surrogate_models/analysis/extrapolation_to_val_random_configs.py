import glob
import json
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from surrogate_models import utils


matplotlib.use("Agg")
from matplotlib import rcParams


rcParams.update({"figure.autolayout": True})


def extrapolation(model_log_dir, nasbench_data, surrogate_model=None):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, "data_config.json"), "r"))
    model_config = json.load(open(os.path.join(model_log_dir, "model_config.json"), "r"))

    if surrogate_model is None:
        # Instantiate model
        surrogate_model = utils.model_dict[model_config["model"]](
            data_root=nasbench_data,
            log_dir=None,
            seed=data_config["seed"],
            data_config=data_config,
            model_config=model_config,
        )
        # Load the model from checkpoint
        surrogate_model.load(os.path.join(model_log_dir, "surrogate_model.model"))

    # Set the same seed as used during the training.
    if data_config["train_val_test_split_random"]["type"] == "no_data":
        random_config_paths = glob.glob(
            os.path.join(
                nasbench_data, "random_fidelities/results_fidelity_*/results_*.json"
            )
        )
    else:
        random_config_paths = json.load(
            open(
                os.path.join(model_log_dir, "train_val_test_split_random_val_paths.json")
            )
        )
    val_results, val_preds, val_target = surrogate_model.evaluate(random_config_paths)

    # Plot the results
    fig = utils.scatter_plot(
        np.array(val_preds),
        np.array(val_target),
        xlabel="Predicted",
        ylabel="True",
        title="Correlation {:5}".format(val_results["kendall_tau"]),
    )
    plt.tight_layout()
    fig.savefig(os.path.join(model_log_dir, "extrapolation_to_random_configs.pdf"))
    plt.close()

    return val_results["kendall_tau"]


@click.command()
@click.option("--model_log_dir", type=click.STRING, help="Experiment directory")
@click.option(
    "--nasbench_data", type=click.STRING, help="Path to nasbench root directory"
)
@click.option("--surrogate_model", default=None)
def extrapolation_interface(model_log_dir, nasbench_data, surrogate_model):
    return extrapolation(model_log_dir, nasbench_data, surrogate_model)


if __name__ == "__main__":
    extrapolation_interface()
