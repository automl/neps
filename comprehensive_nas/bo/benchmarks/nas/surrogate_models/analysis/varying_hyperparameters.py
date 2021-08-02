import glob
import json
import os
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathvalidate import sanitize_filename
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
def hyperparameter_analysis(model_log_dir, nasbench_data):
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
    hyperparameter_change_dict = {
        "OptimizerSelector:sgd:learning_rate": "groundtruths/hyperparameters/learning_rate/",
        "OptimizerSelector:sgd:weight_decay": "groundtruths/hyperparameters/weight_decay/",
    }

    # Get the groundtruth
    matching_files = lambda dir: [
        str(path) for path in Path(os.path.join(nasbench_data, dir)).rglob("*.json")
    ]

    for param_to_change, path in hyperparameter_change_dict.items():
        # Get groundtruths
        groundtruths_paths = matching_files(path)
        groundtruth_data = [json.load(open(path, "r")) for path in groundtruths_paths]

        groundtruth_points = [
            [
                data["optimized_hyperparamater_config"][param_to_change],
                1 - data["info"][0]["val_accuracy_final"] / 100,
            ]
            for data in groundtruth_data
        ]
        param, validation_errors = zip(*groundtruth_points)
        groundtruth_fidelity = {"1": (param, validation_errors)}

        # Get predictions
        fidelity_dict = {}
        for fidelity_num in range(1, 7):
            data_paths = extract_paths_for_budget(val_diagonal_paths, fidelity_num)
            print(fidelity_num, len(data_paths))
            param_dict = {}

            for exponent in np.arange(-1, 6, 0.5):
                param = 10 ** (-exponent)
                surrogate_model.config_loader.parameter_change_dict = {
                    param_to_change: param
                }
                _, val_preds, _ = surrogate_model.evaluate(data_paths)
                val_preds = 1 - np.array(val_preds) / 100
                param_dict[param] = [np.mean(val_preds), np.std(val_preds)]

            fidelity_dict[fidelity_num] = param_dict
            # Plot the results
            plt.figure()
            learning_rates = list(param_dict.keys())
            mean_std = np.array(list(param_dict.values()))
            mean = mean_std[:, 0]
            std = mean_std[:, 1]

            plt.plot(learning_rates, mean, label="Prediction")
            plt.fill_between(learning_rates, mean - std, mean + std, alpha=0.3)

            if groundtruth_fidelity.get(str(fidelity_num)) is not None:
                param, validation_errors = groundtruth_fidelity[str(fidelity_num)]
                plt.scatter(
                    param,
                    validation_errors,
                    label="Groundtruth Fidelity {}".format(fidelity_num),
                )

            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.legend()

            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.xlabel("{}".format(param_to_change))
            plt.ylabel("Validation Error")
            plt.title("Fidelity {}".format(fidelity_num))
            plt.savefig(
                os.path.join(
                    model_log_dir,
                    "high_{}_fidelity_{}.pdf".format(param_to_change, fidelity_num),
                )
            )
            plt.close()

        # Plot subset of the data in one plot
        plt.figure(figsize=(4, 3))
        for fidelity in [1, 3, 5]:
            hyperparameter_values = list(fidelity_dict[fidelity].keys())

            means_stds = np.array(list(fidelity_dict[fidelity].values()))
            means = means_stds[:, 0]
            stds = means_stds[:, 1]
            plt.plot(hyperparameter_values, means, label="Fidelity {}".format(fidelity))
            plt.fill_between(hyperparameter_values, means - stds, means + stds, alpha=0.3)

            if groundtruth_fidelity.get(str(fidelity)) is not None:
                learning_rates, validation_errors = groundtruth_fidelity[str(fidelity)]
                plt.scatter(
                    learning_rates,
                    validation_errors,
                    label="Groundtruth Fidelity {}".format(fidelity),
                )

        plt.xlim(min(hyperparameter_values), max(hyperparameter_values))
        ax = plt.gca()
        ax.set_yscale("log")
        ax = plt.gca()
        ax.set_xscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        plt.xlabel("{}".format(param_to_change))
        plt.ylabel("Validation Error")
        plt.title("Hyperparameter Change")
        plt.tight_layout()
        file_path = os.path.join(
            model_log_dir,
            sanitize_filename("hyperparameter_change_{}.pdf".format(param_to_change)),
        )
        plt.savefig(file_path)
        plt.close()


if __name__ == "__main__":
    hyperparameter_analysis()
