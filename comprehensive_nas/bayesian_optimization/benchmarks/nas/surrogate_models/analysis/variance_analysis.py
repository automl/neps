import copy
import json
import os
import random

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace.read_and_write import json as config_space_json_r_w
from scipy import stats
from surrogate_models import utils
from surrogate_models.analysis.utils import ConfigDict
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_data_paths(surrogate_model):
    return (
        surrogate_model.train_paths,
        surrogate_model.val_paths,
        surrogate_model.test_paths,
    )


def load_data(surrogate_model, result_paths):
    hyps, val_accuracies, test_accuracies = [], [], []

    # query config loader
    for result_path in tqdm(result_paths):
        (
            config_space_instance,
            val_accuracy,
            test_accuracy,
            _,
        ) = surrogate_model.config_loader[result_path]
        hyps.append(config_space_instance.get_dictionary())
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

    return hyps, val_accuracies, test_accuracies


def load_all_data(surrogate_model, train_paths, val_paths, test_paths):
    paths_list = [train_paths, val_paths, test_paths]
    X, y, test_accuracies = [], [], []

    for paths in paths_list:
        X_temp, y_temp, t_temp = load_data(surrogate_model, paths)
        X += X_temp
        y += y_temp
        test_accuracies += t_temp

    return X, y, test_accuracies


def is_in_epsball(new_acc, old_accs, eps=0.05):
    for acc in old_accs:
        if acc > acc - eps and acc < acc + eps:
            True
    return False


def get_fidelity_configs_and_accuracies(config_dict, upper_fidelity, lower_fidelity):
    # get configs in both fidelities
    upper_configs = []
    lower_configs = []
    upper_accuracies = []
    lower_accuracies = []
    for config_hash, accuracy_dict in config_dict.data.items():
        if upper_fidelity in accuracy_dict and lower_fidelity in accuracy_dict:

            # if not is_in_epsball(accuracy_dict[upper_fidelity], upper_accuracies):
            if True:
                upper_config, upper_accuracy = accuracy_dict[upper_fidelity][0]
                lower_config, lower_accuracy = accuracy_dict[lower_fidelity][0]

                upper_configs.append(upper_config)
                upper_accuracies.append(upper_accuracy)
                lower_configs.append(lower_config)
                lower_accuracies.append(lower_accuracy)

    return upper_configs, upper_accuracies, lower_configs, lower_accuracies


def create_random_configs(configspace, n_configs):
    seed_everything(seed=1)

    configs = []
    for ind in range(n_configs):
        configspace.seed(ind)
        config = configspace.sample_configuration().get_dictionary()
        configs.append(config)
    return configs


def change_config_fidelity(config, target_fidelity):
    epoch_fidelities = [50, 88, 155, 274, 483, 851, 1500]
    channel_fidelities = [8, 11, 15, 20, 27, 37, 50]
    cell_fiedelities = [5, 6, 8, 10, 13, 16, 20]

    config[
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max"
    ] = epoch_fidelities[target_fidelity]
    config["NetworkSelectorDatasetInfo:darts:layers"] = cell_fiedelities[target_fidelity]
    config["NetworkSelectorDatasetInfo:darts:init_channels"] = channel_fidelities[
        target_fidelity
    ]
    config["epochs"] = epoch_fidelities[target_fidelity]

    return config


def change_all_config_fidelities(configs, target_fidelity):
    return [change_config_fidelity(config, target_fidelity) for config in configs]


def read_configspace(path):
    with open(path, "r") as f:
        jason_string = f.read()
        configspace = config_space_json_r_w.read(jason_string)
    return configspace


def matrix_plot(matrix, plot_log_dir):
    fig, ax = plt.subplots()

    cax = ax.matshow(matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_yticks(range(7))
    ax.set_xticks(range(7))

    ax.set_title("Rank correlation of the accuracy across fidelities")

    for i in range(7):
        for j in range(7):
            if i == j:
                plt.text(
                    j,
                    i,
                    "{:.2f}".format(1),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:.2f}".format(matrix[i, j]),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    file_path = os.path.join(plot_log_dir)
    plt.savefig(file_path)
    plt.close()


def plot_correlations(
    corr_matrix_true, corr_matrix_model, corr_matrix_random, model_type, model_log_dir
):
    matrix_plot(
        corr_matrix_true,
        plot_log_dir=os.path.join(
            model_log_dir, "fidelity_correlation_true_" + str(model_type) + ".pdf"
        ),
    )
    matrix_plot(
        corr_matrix_model,
        plot_log_dir=os.path.join(
            model_log_dir, "fidelity_correlation_model_noisy_" + str(model_type) + ".pdf"
        ),
    )
    matrix_plot(
        corr_matrix_random,
        plot_log_dir=os.path.join(
            model_log_dir, "fidelity_correlation_random_noisy_" + str(model_type) + ".pdf"
        ),
    )


def get_variances(config_dict):
    fidelity_variances = []

    for fid in range(7):
        print("Calculating variance for fidelity %i" % fid)
        fidelity_stds = []
        for config_hash, subdict in config_dict.data.items():
            if fid in subdict.keys() and len(subdict[fid]) > 1:
                accuracies = [a for c, a in subdict[fid]]
                fidelity_stds.append(np.std(accuracies))
        fidelity_variances.append(np.mean(fidelity_stds))
    return fidelity_variances


def query_with_noise(surrogate_model, config, variances, fidelity):
    query_results = surrogate_model.query(config)
    noise = np.random.normal(1, variances[fidelity], 1)
    return query_results + noise


def fidelity_correlation(model_log_dir, nasbench_data, configspace_path):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, "data_config.json"), "r"))
    model_config = json.load(open(os.path.join(model_log_dir, "model_config.json"), "r"))

    # Instantiate model
    surrogate_model = utils.model_dict[model_config["model"]](
        data_root=nasbench_data,
        log_dir=model_log_dir,
        seed=data_config["seed"],
        data_config=data_config,
        model_config=model_config,
    )

    # Load the model from checkpoint
    print("==> Loading surrogate model")
    surrogate_model.load(os.path.join(model_log_dir, "surrogate_model.model"))

    # Load result paths
    print("==> Loading data")
    train_paths, val_paths, test_paths = get_data_paths(surrogate_model)

    # Load data
    if False:
        train_paths = train_paths[0:100]
        val_paths = val_paths[0:10]
        test_paths = val_paths[0:10]
    X, y, _ = load_all_data(surrogate_model, train_paths, val_paths, test_paths)

    # Hash configs
    print("==> Creating hashes")
    config_hashdict = ConfigDict(
        X, y, paths=None, include_off_diagonal=False, allow_repeats=True
    )

    variances = get_variances(config_hashdict)
    print(variances)

    corr_matrix_true = np.zeros((7, 7))
    corr_matrix_model = np.zeros((7, 7))
    corr_matrix_random = np.zeros((7, 7))

    # Get correlations
    for start_from_fidelity_fidelity in range(7):
        real_corrs, model_corrs, model_simulated_corrs = [], [], []

        for fidelity in range(7):
            if fidelity == start_from_fidelity_fidelity:
                continue

            print("==> Fidelity %i" % fidelity)

            # Correlation on data
            (
                upper_configs,
                upper_accuracies,
                lower_configs,
                lower_accuracies,
            ) = get_fidelity_configs_and_accuracies(
                config_hashdict, start_from_fidelity_fidelity, fidelity
            )
            real_corr, p_coeff = stats.kendalltau(upper_accuracies, lower_accuracies)
            real_corrs.append(real_corr)

            # Get correlation on model(data)
            model_upper_pred = [
                query_with_noise(
                    surrogate_model, config, variances, start_from_fidelity_fidelity
                )
                for config in upper_configs
            ]
            model_lower_pred = [
                query_with_noise(surrogate_model, config, variances, fidelity)
                for config in lower_configs
            ]
            model_corr, p_coeff = stats.kendalltau(model_upper_pred, model_lower_pred)
            model_corrs.append(model_corr)

            # Get correlation on artificial data
            configspace = read_configspace(configspace_path)
            simulated_configs = create_random_configs(configspace, 100)
            simulated_upper_configs = change_all_config_fidelities(
                copy.deepcopy(simulated_configs), start_from_fidelity_fidelity
            )
            simulated_lower_configs = change_all_config_fidelities(
                copy.deepcopy(simulated_configs), fidelity
            )
            model_simulated_upper_pred = [
                query_with_noise(
                    surrogate_model, config, variances, start_from_fidelity_fidelity
                )
                for config in simulated_upper_configs
            ]
            model_simulated_lower_pred = [
                query_with_noise(surrogate_model, config, variances, fidelity)
                for config in simulated_lower_configs
            ]
            model_simulated_corr, p_coeff = stats.kendalltau(
                model_simulated_upper_pred, model_simulated_lower_pred
            )
            model_simulated_corrs.append(model_simulated_corr)

            corr_matrix_true[start_from_fidelity_fidelity, fidelity] = real_corr
            corr_matrix_model[start_from_fidelity_fidelity, fidelity] = model_corr
            corr_matrix_random[
                start_from_fidelity_fidelity, fidelity
            ] = model_simulated_corr
            print(
                "From %i to %i: (%f, %f, %f)"
                % (
                    start_from_fidelity_fidelity,
                    fidelity,
                    real_corr,
                    model_corr,
                    model_simulated_corr,
                )
            )

    plot_correlations(
        corr_matrix_true,
        corr_matrix_model,
        corr_matrix_random,
        model_config["model"],
        model_log_dir,
    )

    return real_corrs, model_corrs, model_simulated_corrs


@click.command()
@click.option("--model_log_dir", type=click.STRING, help="Experiment directory")
@click.option(
    "--nasbench_data", type=click.STRING, help="Path to nasbench root directory"
)
@click.option("--configspace_path", type=click.STRING, help="Path to configspace json")
def fidelity_correlation_interface(model_log_dir, nasbench_data, configspace_path):
    return fidelity_correlation(model_log_dir, nasbench_data, configspace_path)


if __name__ == "__main__":
    fidelity_correlation_interface()
