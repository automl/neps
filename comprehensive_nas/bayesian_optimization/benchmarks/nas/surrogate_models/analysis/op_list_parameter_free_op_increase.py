import glob
import json
import os
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from surrogate_models import utils
from surrogate_models.analysis.utils import extract_paths_for_budget

matplotlib.use("Agg")

rcParams.update({"figure.autolayout": True})


@click.command()
@click.option("--model_log_dir", type=click.STRING, help="Experiment directory")
@click.option(
    "--nasbench_data", type=click.STRING, help="Path to nasbench root directory"
)
def op_list_skip_connect_increase(model_log_dir, nasbench_data):
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
    matching_files = lambda dir: [
        str(path) for path in Path(os.path.join(nasbench_data, dir)).rglob("*.json")
    ]

    # Take all datapoints in a diagonal fidelity and transform them to the other fidelities.
    for fidelity_num in range(0, 7):
        data_paths = extract_paths_for_budget(val_diagonal_paths, fidelity_num)
        print(fidelity_num, len(data_paths))
        # Iterate through the parameter types
        ratio_skip_connection_in_cell_dict = {
            "max_pool_3x3": {},
            "avg_pool_3x3": {},
            "skip_connect": {},
        }
        for parameter_free_op in ratio_skip_connection_in_cell_dict.keys():
            surrogate_model.config_loader.parameter_free_op_increase_type = (
                parameter_free_op
            )
            # Progressively increase the number of the selected parameter free operation
            for ratio_parameter_free_op_in_cell in np.arange(0.0, 1.05, 0.1):
                surrogate_model.config_loader.ratio_parameter_free_op_in_cell = (
                    ratio_parameter_free_op_in_cell
                )
                val_pred_results = []
                for i in range(4):
                    _, val_preds, _ = surrogate_model.evaluate(data_paths)
                    val_pred_results.extend(val_preds)
                ratio_skip_connection_in_cell_dict[parameter_free_op][
                    ratio_parameter_free_op_in_cell
                ] = [np.mean(val_pred_results), np.std(val_pred_results)]

        # Plot the results
        plt.figure(figsize=(4, 3))
        for op_type, ratio_data in ratio_skip_connection_in_cell_dict.items():
            ratios = list(ratio_data.keys())
            mean_std = np.array(list(ratio_data.values()))
            mean = mean_std[:, 0]
            std = mean_std[:, 1]

            plt.plot(ratios, mean, label=op_type)
            plt.fill_between(ratios, mean - std, mean + std, alpha=0.3)

        gt_data = matching_files(
            os.path.join(
                nasbench_data,
                "groundtruths/low_parameter/results_fidelity_{}".format(fidelity_num),
            )
        )

        if len(gt_data) != 0:
            groundtruth_points = [
                json.load(open(data, "r"))["info"][0]["val_accuracy_final"]
                for data in gt_data
            ]
            x_axis = np.ones_like(groundtruth_points)
            plt.plot(
                x_axis,
                groundtruth_points,
                label="Groundtruth Fidelity {}".format(fidelity_num),
            )

        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel("Ratio parameter free op in cell".format())
        plt.ylabel("Validation Accuracy")
        plt.title("Fidelity {}".format(fidelity_num))
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                model_log_dir,
                "parameter_free_op_ratio_in_fidelity_{}.pdf".format(fidelity_num),
            )
        )
        plt.close()


if __name__ == "__main__":
    op_list_skip_connect_increase()
