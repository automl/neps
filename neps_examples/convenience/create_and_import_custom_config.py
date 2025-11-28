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
    operation1 = neps.Categorical(
        choices=[
            "option1",
            "option2",
            neps.Operation(
                operator="option3",
                args=(float1, cat1.resample()),
                kwargs={"param1": float1.resample()},
            ),
        ]
    )


if __name__ == "__main__":
    # We create a configuration interactively and receive both
    # the configuration dictionary and a dictionary of the sampled parameters.
    config, pipeline = neps.create_config(ExampleSpace())
    print("Created configuration:")
    pprint(config)

    logging.basicConfig(level=logging.INFO)
    # The created configuration can then be used as an imported trial in NePS optimizers.
    # We demonstrate this with the fictional result of objective_to_minimize = 0.5
    neps.import_trials(
        evaluated_trials=[(config, neps.UserResultDict(objective_to_minimize=0.5))],
        root_directory="results/created_config_example",
        pipeline_space=ExampleSpace(),
        overwrite_root_directory=True,
    )
