import copy
import glob
import json
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathvalidate import sanitize_filename
from surrogate_models import utils
from surrogate_models.analysis.utils import extract_paths_for_budget
from tqdm import tqdm


matplotlib.use("Agg")
from matplotlib import rcParams


rcParams.update({"figure.autolayout": True})


def find_key_value(key, dictionary):
    """
    Check if key is contained in dictionary in a nested way
    Source: https://gist.github.com/douglasmiranda/5127251#file-gistfile1-py-L2
    :param key:
    :param dictionary:
    :return:
    """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find_key_value(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find_key_value(key, d):
                    yield result


def filter_edge_results(nasbench_data, result_filter):
    config_loader = utils.ConfigLoader(config_space_path="configspace.json")
    all_results_paths = glob.glob(
        os.path.join(
            nasbench_data, "groundtruths/edges/results_fidelity_*/results_*.json"
        )
    )
    results = []
    for result_path in tqdm(all_results_paths, desc="Filtering results"):
        config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[
            result_path
        ]
        result_json = config_space_instance.get_dictionary()
        filtered_result = []
        # Go through all elements to be filtered
        for filter_key, filter_details in result_filter.items():
            # Retrieve the element to be checked
            filtered_value = list(find_key_value(filter_key, result_json))
            if len(filtered_value):
                if filter_details["type"] == "interval":
                    # Check if the configuration matches the filter interval
                    lower_filter_val, high_filter_val = filter_details["data"]
                    if lower_filter_val <= filtered_value[0] <= high_filter_val:
                        filtered_result.append(result_path)
                    else:
                        continue
                elif filter_details["type"] == "list":
                    # Check whether the value is in a list of pre-specified values
                    if filtered_value[0] in filter_details["data"]:
                        filtered_result.append(result_path)
                    else:
                        continue
                else:
                    pass
        # Did the path fulfill all filters?
        if len(filtered_result) == len(result_filter.keys()):
            results.append(filtered_result[0])
    return results


def load_gt(param_to_change, groundtruths_paths):
    config_loader = utils.ConfigLoader(config_space_path="configspace.json")
    groundtruth_data_config = [
        config_loader[path][0].get_dictionary() for path in groundtruths_paths
    ]
    groundtruth_json_file = [config_loader[path][3] for path in groundtruths_paths]

    groundtruth_points = [
        [
            data_config[param_to_change],
            1 - json_data["info"][0]["val_accuracy_final"] / 100,
        ]
        for data_config, json_data in zip(groundtruth_data_config, groundtruth_json_file)
    ]
    param, validation_errors = zip(*groundtruth_points)
    groundtruth_fidelity = {"0": (param, validation_errors)}
    return groundtruth_fidelity


@click.command()
@click.option("--model_log_dir", type=click.STRING, help="Experiment directory")
@click.option(
    "--nasbench_data", type=click.STRING, help="Path to nasbench root directory"
)
def fidelity_analysis(model_log_dir, nasbench_data):
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

    parameter_change_list = surrogate_model.config_loader.fidelity_multiplier

    result_filter = {
        "epochs": {"type": "list", "data": [50]},
        "NetworkSelectorDatasetInfo:darts:init_channels": {"type": "list", "data": [8]},
        "NetworkSelectorDatasetInfo:darts:layers": {"type": "list", "data": [5]},
    }
    for (param_to_change, multiplier), init_value in zip(
        parameter_change_list.items(), [50, 8, 5]
    ):
        result_filter_copy = copy.deepcopy(result_filter)
        result_filter_copy.pop(param_to_change)
        gt_varied_fidelity = filter_edge_results(nasbench_data, result_filter_copy)
        groundtruth_fidelity = load_gt(param_to_change, gt_varied_fidelity)
        fidelity_dict = {}
        for fidelity_num in range(0, 1):
            data_paths = extract_paths_for_budget(val_diagonal_paths, fidelity_num)
            print(fidelity_num, len(data_paths))
            param_dict = {}

            for exponent in np.arange(-1, 9, 0.5):
                param = init_value * multiplier ** (exponent)
                surrogate_model.config_loader.parameter_change_dict = {
                    param_to_change: int(param)
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

            plt.plot(learning_rates, mean)
            plt.fill_between(learning_rates, mean - std, mean + std, alpha=0.3)

            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")

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
        fig = plt.figure(figsize=(4, 3))
        for fidelity in [0]:
            hyperparameter_values = list(fidelity_dict[fidelity].keys())

            means_stds = np.array(list(fidelity_dict[fidelity].values()))
            means = means_stds[:, 0]
            stds = means_stds[:, 1]
            plt.plot(hyperparameter_values, means, label="Fidelity {}".format(fidelity))
            plt.fill_between(hyperparameter_values, means - stds, means + stds, alpha=0.3)

        plt.axvline(x=init_value)
        param, validation_errors = groundtruth_fidelity[str(0)]
        plt.scatter(param, validation_errors, label="Groundtruth Fidelity {}".format(0))
        fig.autofmt_xdate()
        plt.xlim(min(hyperparameter_values), max(hyperparameter_values))
        ax = plt.gca()
        ax.set_yscale("log")
        ax = plt.gca()
        ax.set_xscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.xlabel("{}".format(param_to_change))
        plt.ylabel("Validation Error")
        plt.title("Fidelity Change")
        plt.tight_layout()
        file_path = os.path.join(
            model_log_dir,
            sanitize_filename("fidelity_change_{}.pdf".format(param_to_change)),
        )
        plt.savefig(file_path)
        plt.close()


if __name__ == "__main__":
    fidelity_analysis()
