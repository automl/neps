import json
import os
import time

import click
import matplotlib
import scipy.stats as stats
import sklearn
import yaml
from surrogate_models import utils

matplotlib.use("Agg")


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(utils.model_dict.keys())),
    default="gnn",
    help="which surrogate model to fit",
)
@click.option("--darts_data", type=click.STRING, help="path to nasbench root directory")
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
    "--model_path",
    type=click.STRING,
    help="path to saved surrogate model.",
    default="experiments/surrogate_models/lgb/20200130-153015-6/surrogate_model.model",
)
@click.option(
    "--log_dir",
    type=click.STRING,
    help="Experiment directory",
    default="experiments/darts_test",
)
@click.option("--seed", type=click.INT, help="seed for numpy, python, pytorch", default=6)
def train_surrogate_model(
    model, darts_data, model_config_path, model_path, data_config_path, log_dir, seed
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
        data_root="None",
        log_dir=None,
        seed=seed,
        model_config=model_config,
        data_config=data_config,
    )

    # Load model from checkpoint
    surrogate_model.load(os.path.join(model_path, "surrogate_model.model"))

    # load darts test data
    with open(os.path.join(darts_data, "evals.yaml")) as f:
        evals = yaml.load(f)

    test_pred, y_test = [], []
    for idx in [10473, 10475, 10477, 10478, 12267, 12270, 12273, 12276]:
        for seed in range(3):
            config_dict = json.load(
                open(
                    os.path.join(
                        darts_data, "configs", "config_%d_%d.json" % (idx, seed)
                    ),
                    "r",
                )
            )
            config_dict["SimpleTrainNode:mixup:alpha"] = 0.2
            config_dict["epochs"] = 600

            pred = surrogate_model.query(config_dict=config_dict)[0]
            test_pred.append(pred)
            y_test.append(100 - evals[idx][seed])

            print("==> Settings: %d %d" % (idx, seed))
            print("Query result", pred)
            print("True result", evals[idx][seed])

    utils.plot_predictions(
        test_pred,
        test_pred,
        None,
        None,
        y_test,
        y_test,
        log_dir=log_dir,
        name="%s darts configs" % (model),
        x1=90,
        y1=90,
    )

    print("Pearson: ", stats.pearsonr(test_pred, y_test))
    print("Spearman: ", stats.spearmanr(test_pred, y_test))
    print("Kendall: ", stats.kendalltau(test_pred, y_test))
    print("R2: ", sklearn.metrics.r2_score(test_pred, y_test))

    # Test the model
    # surrogate_model.test()


if __name__ == "__main__":
    train_surrogate_model()
