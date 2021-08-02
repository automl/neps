import glob
import json
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from surrogate_models import utils
from surrogate_models.analysis.utils import extract_paths_for_budget


matplotlib.use("Agg")
from matplotlib import rcParams


rcParams.update({"figure.autolayout": True})


@click.command()
@click.option("--model_log_dir", type=click.STRING, help="Experiment directory")
@click.option(
    "--nasbench_data", type=click.STRING, help="Path to nasbench root directory"
)
@click.option(
    "--last_fidelity",
    type=click.INT,
    help="Until which fidelity (inc.) to go.",
    default=6,
)
def interpolation_analysis(model_log_dir, nasbench_data, last_fidelity):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, "data_config.json"), "r"))
    model_config = json.load(open(os.path.join(model_log_dir, "model_config.json"), "r"))

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
    np.random.seed(data_config["seed"])

    # Set the same seed as used during the training.
    if data_config["train_val_test_split_diagonal"]["type"] == "no_data":
        val_diagonal_paths = glob.glob(
            os.path.join(nasbench_data, "run_*/results_fidelity_*/results_*.json")
        )
    else:
        val_diagonal_paths = json.load(
            open(
                os.path.join(
                    model_log_dir, "train_val_test_split_diagonal_val_paths.json"
                )
            )
        )

    epoch_dict = {fidelity_num: {} for fidelity_num in range(last_fidelity + 1)}

    # Take all datapoints in a diagonal fidelity and transform them to the other fidelities.
    for fidelity_num in range(0, last_fidelity + 1):
        data_paths = extract_paths_for_budget(val_diagonal_paths, fidelity_num)
        print(fidelity_num, len(data_paths))
        for fidelity in range(-fidelity_num, last_fidelity + 1 - fidelity_num):
            surrogate_model.config_loader.fidelity_exponent = fidelity
            _, val_preds, _ = surrogate_model.evaluate(data_paths)
            epoch_dict[fidelity_num][fidelity_num + fidelity] = val_preds

        # Plot the results
        plt.figure()
        for epoch, pred_at_epoch in enumerate(epoch_dict[fidelity_num].values()):
            epoch_x = np.ones_like(pred_at_epoch) * epoch
            plt.scatter(epoch_x, pred_at_epoch)
            ax = plt.gca()
            ax.axvline(fidelity_num)

        plt.xlabel("Fidelity")
        plt.ylabel("Validation Accuracy")
        plt.savefig(
            os.path.join(
                model_log_dir,
                "interpolation_experiment_start_at_fidelity_{}.pdf".format(fidelity_num),
            )
        )
        plt.close()

    # Plot subset of the data in one plot
    plt.figure(figsize=(4, 3))
    for fidelity in [0, 2, 4]:
        fidelities = list(epoch_dict.keys())

        means, stds = [], []  # ;D
        for fidelity_sim, pred_values_at_fidelity_sim in epoch_dict[fidelity].items():
            means.append(np.mean(1 - np.array(pred_values_at_fidelity_sim) / 100))
            stds.append(np.std(1 - np.array(pred_values_at_fidelity_sim) / 100))

        means, stds = np.array(means), np.array(stds)
        plt.plot(fidelities, means, label="Fidelity {}".format(fidelity))
        plt.fill_between(fidelities, means - stds, means + stds, alpha=0.3)

    plt.xlim(min(fidelities), max(fidelities))
    ax = plt.gca()
    ax.set_yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlabel("Fidelity")
    plt.ylabel("Validation Error")
    plt.title("Interpolation")
    plt.tight_layout()
    plt.savefig(os.path.join(model_log_dir, "interpolation.pdf"))
    plt.close()


if __name__ == "__main__":
    interpolation_analysis()
