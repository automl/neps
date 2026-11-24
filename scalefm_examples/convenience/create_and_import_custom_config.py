"""How to create an ScaleFM configuration manually which can then be used as imported trial."""

import scalefm
from pprint import pprint
import logging


# This example space demonstrates all types of parameters available in ScaleFM.
class ExampleSpace(scalefm.PipelineSpace):
    int1 = scalefm.IntegerFidelity(1, 10)
    float1 = scalefm.Float(0.0, 1.0)
    cat1 = scalefm.Categorical(["a", "b", "c"])
    cat2 = scalefm.Categorical(["x", "y", float1])
    operation1 = scalefm.Categorical(
        choices=[
            "option1",
            "option2",
            scalefm.Operation(
                operator="option3",
                args=(float1, cat1.resample()),
                kwargs={"param1": float1.resample()},
            ),
        ]
    )


if __name__ == "__main__":
    # We create a configuration interactively and receive both
    # the configuration dictionary and a dictionary of the sampled parameters.
    config, pipeline = scalefm.create_config(ExampleSpace())
    print("Created configuration:")
    pprint(config)

    logging.basicConfig(level=logging.INFO)
    # The created configuration can then be used as an imported trial in ScaleFM optimizers.
    # We demonstrate this with the fictional result of objective_to_minimize = 0.5
    scalefm.import_trials(
        evaluated_trials=[(config, scalefm.UserResultDict(objective_to_minimize=0.5, cost=1.0))],
        root_directory="results/created_config_example",
        pipeline_space=ExampleSpace(),
        overwrite_root_directory=True,
    )
