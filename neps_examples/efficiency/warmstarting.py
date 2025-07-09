import neps
import logging
from neps import Pipeline, Integer, Float, Fidelity
from neps.space.neps_spaces import neps_space


class SimpleSpace(Pipeline):
    int_param = Integer(0, 10)
    float_param = Float(0.0, 1.0)
    epochs = Fidelity(Integer(1, 5))


# Sampling a random configuration of the pipeline, which will be used for warmstarting
pipeline = SimpleSpace()
resolved_pipeline, resolution_context = neps_space.resolve(
    pipeline, environment_values={"epochs": 5}
)


def evaluate_pipeline(int_param, float_param, epochs=5):
    # This is a dummy evaluation function that just returns the weighted sum
    return {"objective_to_minimize": (int_param + float_param)*epochs, "cost": epochs}


wanted_config = resolution_context.samplings_made
wanted_env = resolution_context.environment_values
wanted_result = evaluate_pipeline(**resolved_pipeline.get_attrs())
warmstarting_configs = [(wanted_config, wanted_env, wanted_result)]


logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=SimpleSpace(),
    root_directory="results/warmstart_example/",
    max_evaluations_total=15,
    optimizer=neps.algorithms.neps_priorband,
    warmstart_configs=warmstarting_configs
)
