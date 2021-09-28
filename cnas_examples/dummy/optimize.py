import comprehensive_nas as cnas


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    config = dict(operant_a=10, operant_b=100, operation="add")  # Remove once dummy works

    if config["operation"] == "multiply":
        return config["operant_a"] * config["operant_b"]
    elif config["operation"] == "add":
        return config["operant_a"] + config["operant_b"]
    else:
        raise ValueError


if __name__ == "__main__":
    pipeline_space = cnas.PipelineSpace(
        operation=cnas.Categorical(choices={"multiply", "add"}),
        operant_a=cnas.Integer(lower=1, upper=100),
        operant_b=cnas.Float(lower=1, upper=100, log=True),
        architecture=cnas.DenseGraph(num_nodes=3, edge_choices={"identity", "3x3_conv"}),
    )
    cnas.run_comprehensive_nas(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/dummy_example",
        n_iterations=10,
        searcher="dummy_random",
        overwrite_logging=True,
    )
