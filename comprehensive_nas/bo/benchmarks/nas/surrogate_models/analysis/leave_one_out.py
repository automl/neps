import json
import os
import random
import time

import click
import matplotlib
import numpy as np

from ConfigSpace.read_and_write import json as config_space_json_r_w
from sklearn.model_selection import train_test_split
from surrogate_models import utils
from surrogate_models.analysis.utils import ConfigDict


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
    for result_path in result_paths:
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
                upper_config, upper_accuracy = accuracy_dict[upper_fidelity]
                lower_config, lower_accuracy = accuracy_dict[lower_fidelity]

                upper_configs.append(upper_config)
                upper_accuracies.append(upper_accuracy)
                lower_configs.append(lower_config)
                lower_accuracies.append(lower_accuracy)

    return upper_configs, upper_accuracies, lower_configs, lower_accuracies


def read_configspace(path):
    with open(path, "r") as f:
        jason_string = f.read()
        configspace = config_space_json_r_w.read(jason_string)
    return configspace


def split_leaving_one_out(config_hashdict, leave_out_fidelity, val_split=0.2):
    # get configs which are evaluated at left_out_fidelity
    leave_out_hashes = []
    train_hashes = []
    for config_hash, accuracy_dict in config_hashdict.data.items():
        if 4 in accuracy_dict.keys():
            leave_out_hashes.append(config_hash)
        else:
            train_hashes.append(config_hash)

    # get test paths:
    test_paths = [config_hashdict.data[h][4][-1] for h in leave_out_hashes]
    print("==> Found %i configs on left out fidelity" % len(test_paths))

    # get val train paths:
    n_configs_on_higher_fidelities = 0
    train_val_paths = []
    for h in train_hashes:
        for fidelity, (config, accuracy, path) in config_hashdict.data[h].items():
            if fidelity > leave_out_fidelity:
                n_configs_on_higher_fidelities += 1
            train_val_paths.append(path)

    print(
        "==> Found %i configs on fidelities higher then left out fidelity"
        % n_configs_on_higher_fidelities
    )

    train_paths, val_paths = train_test_split(
        train_val_paths, test_size=val_split, random_state=42
    )

    return train_paths, val_paths, test_paths


def leave_one_out(
    model,
    nasbench_data,
    model_config_path,
    data_config_path,
    log_dir,
    seed,
    leave_out_fidelity,
):
    # Load config
    data_config = json.load(open(data_config_path, "r"))
    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)
        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, "r"))
    model_config["model"] = model

    # Create log directory
    log_dir = os.path.join(
        log_dir, model, "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), seed)
    )
    os.makedirs(log_dir)

    # Instantiate model
    surrogate_model = utils.model_dict[model](
        data_root=nasbench_data,
        log_dir=log_dir,
        seed=seed,
        model_config=model_config,
        data_config=data_config,
    )

    # Load result paths
    print("==> Loading data")
    train_paths, val_paths, test_paths = get_data_paths(surrogate_model)
    X, y, _ = load_all_data(surrogate_model, train_paths, val_paths, test_paths)
    all_paths = train_paths + val_paths + test_paths

    # Hash configs
    print("==> Creating hashes")
    config_hashdict = ConfigDict(X, y, all_paths, include_off_diagonal=True)

    # Split data
    new_train_paths, new_val_paths, new_test_paths = split_leaving_one_out(
        config_hashdict, leave_out_fidelity, val_split=0.2
    )

    # Override paths in the model
    surrogate_model.train_paths = new_train_paths
    surrogate_model.val_paths = new_val_paths
    surrogate_model.test_paths = new_test_paths

    # Train
    surrogate_model.train()

    surrogate_model.test()


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(utils.model_dict.keys())),
    default="gnn",
    help="which surrogate model to fit",
)
@click.option(
    "--nasbench_data", type=click.STRING, help="path to nasbench root directory"
)
@click.option(
    "--model_config_path",
    type=click.STRING,
    help="Leave None to use the default config.",
    default=None,
)
@click.option(
    "--data_config_path",
    type=click.STRING,
    help="Path to config.json",
    default="surrogate_models/configs/data_configs/diagonal_plus_off_diagonal.json",
)
@click.option(
    "--log_dir",
    type=click.STRING,
    help="Experiment directory",
    default="experiments/surrogate_models",
)
@click.option("--seed", type=click.INT, help="seed for numpy, python, pytorch", default=6)
@click.option("--leave_out_fidelity", type=click.INT, default=4)
def leave_one_out_interface(
    model,
    nasbench_data,
    model_config_path,
    data_config_path,
    log_dir,
    seed,
    leave_out_fidelity,
):
    return leave_one_out(
        model,
        nasbench_data,
        model_config_path,
        data_config_path,
        log_dir,
        seed,
        leave_out_fidelity,
    )


if __name__ == "__main__":
    leave_one_out_interface()
