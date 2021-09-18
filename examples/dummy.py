import ConfigSpace as CS

from comprehensive_nas.api import run_comprehensive_nas


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    return config["counter"]


if __name__ == "__main__":
    # Needs to be the configspace type expected by the approach, here: random search
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.hyperparameters.UniformIntegerHyperparameter("counter", lower=1, upper=100)
    )
    run_comprehensive_nas(
        run_pipeline=run_pipeline,
        config_space=config_space,
        working_directory="results/dummy_example",
        n_iterations=10,
        searcher="dummy_random",
        overwrite_logging=True,
    )
