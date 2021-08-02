import json
import os
import time

import click
import matplotlib

from surrogate_models import utils


matplotlib.use("Agg")


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
def train_surrogate_model(
    model, nasbench_data, model_config_path, data_config_path, log_dir, seed
):
    # Load config
    data_config = json.load(open(data_config_path, "r"))

    # Create log directory
    log_dir = os.path.join(
        log_dir, model, "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), seed)
    )
    os.makedirs(log_dir)

    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)

        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, "r"))
    model_config["model"] = model

    # Instantiate surrogate model
    surrogate_model = utils.model_dict[model](
        data_root=nasbench_data,
        log_dir=log_dir,
        seed=seed,
        model_config=model_config,
        data_config=data_config,
    )

    # Train and validate the model on the available data
    surrogate_model.train()

    # Test querying
    json_file = json.load(
        open("surrogate_models/test/results_fidelity_0/results_0.json", "r")
    )
    config_dict = json_file["optimized_hyperparamater_config"]
    config_dict["epochs"] = 55
    print("Query result", surrogate_model.query(config_dict=config_dict))

    # Test the model
    surrogate_model.test()

    # Save the model
    surrogate_model.save()


if __name__ == "__main__":
    train_surrogate_model()
