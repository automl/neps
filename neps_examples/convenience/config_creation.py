"""How to create a NePS configuration manually which can then be used as imported trial."""

import neps
from pprint import pprint
import logging


# This example space demonstrates all types of parameters available in NePS.
class ExampleSpace(neps.PipelineSpace):
    int1 = neps.Fidelity(neps.Integer(1, 10))
    float1 = neps.Float(0.0, 1.0)
    cat1 = neps.Categorical(["a", "b", "c"])
    cat2 = neps.Categorical(["x", "y", float1])
    cat4 = neps.Categorical([neps.Resampled(cat2), neps.Resampled(cat1)])


if __name__ == "__main__":
    # We create a configuration interactively and receive both
    # the configuration dictionary and a the corresponding pipeline.
    config, pipeline = neps.create_config(ExampleSpace())
    print("Created configuration:")
    pprint(config)
    print("Sampled pipeline:")
    print(pipeline, "\n")
    # We can access the sampled values via e.g. pipeline.int1

    logging.basicConfig(level=logging.INFO)
    # The created configuration can then be used as an imported trial in NePS optimizers.
    # We demonstrate this with the fictional result of objective_to_minimize = 0.5
    neps.import_trials(
        ExampleSpace(),
        [(config, neps.UserResultDict(objective_to_minimize=0.5))],
        root_directory="results/created_config_example",
        overwrite_root_directory=True,
    )
