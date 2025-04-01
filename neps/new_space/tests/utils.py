import pprint
from typing import Callable

from neps.new_space import space


def generate_possible_config_strings(
    pipeline: space.Pipeline,
    resolved_pipeline_attr_getter: Callable[[space.Pipeline], space.Operation],
    num_resolutions: int = 50_000,
    display: bool = True,
):
    result = set()

    for _ in range(num_resolutions):
        resolved_pipeline, sampled_values = space.resolve(pipeline)
        attr = resolved_pipeline_attr_getter(resolved_pipeline)
        config_string = space.to_config_string(attr)
        result.add(config_string)

    if display:
        pprint.pprint(result, indent=2)

    return result
