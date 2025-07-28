import neps
import logging
from neps import PipelineSpace, Integer, Float, Fidelity, Operation, Categorical
from neps.space.neps_spaces import neps_space


def operation(x):
    """A simple operation that can be used in the pipeline."""
    return x


class SimpleSpace(PipelineSpace):
    int_param = Integer(0, 10)
    float_param = Float(0.0, 1.0)
    epochs = Fidelity(Integer(1, 5))
    # cat_param = Categorical((float_param, int_param))
    # op = Operation(operation, args=(float_param, int_param))
    # op2 = Operation("Test")
    # op3 = Operation("Test2", args=(int_param,))


# Sampling a random configuration of the pipeline, which will be used for warmstarting
pipeline = SimpleSpace()
resolved_pipeline, resolution_context = neps_space.resolve(
    pipeline, environment_values={"epochs": 5}
)
# for operator in (resolved_pipeline.op, resolved_pipeline.op2, resolved_pipeline.op3):
#     print("Resolved Pipeline:", operator)
#     if callable(operator):
#         print("Callable:", neps_space.convert_operation_to_callable(operator).__str__())
#     print(neps_space.convert_operation_to_string(resolved_pipeline.op))
#     print(
#         neps_space.config_string.ConfigString(
#             neps_space.convert_operation_to_string(operator)
#         ).pretty_format()
#     )
#     print(
#         neps_space.config_string.ConfigString(
#             neps_space.convert_operation_to_string(operator)
#         )
#     )
#     print(
#         neps_space.config_string.ConfigString(
#             neps_space.convert_operation_to_string(operator)
#         ).unwrapped
#     )


def evaluate_pipeline(int_param, float_param, epochs=5, **kwargs) -> dict[str, float]:
    return {"objective_to_minimize": -(int_param + float_param) * epochs, "cost": epochs}


wanted_config = resolution_context.samplings_made
wanted_env = resolution_context.environment_values
wanted_result = evaluate_pipeline(**resolved_pipeline.get_attrs())
warmstarting_configs = [
    (wanted_config, wanted_env, wanted_result),
    # (wanted_config, {"epochs": 2}, wanted_result),
]

from functools import partial

# Running the NEPS pipeline with warmstarting
logging.basicConfig(level=logging.INFO)
neps.warmstart_neps(
    pipeline,
    "results/warmstart_example/",
    warmstarting_configs,
    overwrite_working_directory=True,
    optimizer=partial(
        neps.algorithms.neps_random_search,
        use_priors=True,
        ignore_fidelity="highest fidelity",
    ),
)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=SimpleSpace(),
    root_directory="results/warmstart_example/",
    max_evaluations_per_run=5,
    optimizer=partial(
        neps.algorithms.neps_random_search,
        use_priors=True,
        ignore_fidelity="highest fidelity",
    ),
    # warmstart_configs=warmstarting_configs,
    overwrite_working_directory=False,
)
neps.status(
    "results/warmstart_example",
    print_summary=True,
    pipeline_space_variables=(
        SimpleSpace(),
        ["int_param", "float_param", "epochs"],  # , "op", "op2", "op3", "cat_param"],
    ),
)
