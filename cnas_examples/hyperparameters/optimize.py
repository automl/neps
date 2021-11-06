import comprehensive_nas as cnas


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    if config["operation"] == "multiply":
        return config["operant_a"] * config["operant_b"]
    elif config["operation"] == "add":
        return config["operant_a"] + config["operant_b"]
    else:
        raise ValueError


if __name__ == "__main__":
    # We want this API at some point:
    # pipeline_space = cnas.SearchSpace(
    #     operation=cnas.Categorical(choices={"multiply", "add"}),
    #     operant_a=cnas.Integer(lower=1, upper=100),
    #     operant_b=cnas.Float(lower=1, upper=100, log=True),
    # )
    pipeline_space = cnas.SearchSpace(
        cnas.CategoricalHyperparameter(name="operation", choices={"multiply", "add"}),
        cnas.IntegerHyperparameter(name="operant_a", lower=1, upper=100),
        cnas.FloatHyperparameter(name="operant_b", lower=1, upper=100, log=True),
    )
    cnas.run_comprehensive_nas(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/dummy_example",
        n_iterations=10,
        searcher="bayesian_optimization",
        overwrite_logging=True,
    )
