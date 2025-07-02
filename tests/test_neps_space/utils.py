from __future__ import annotations

from collections.abc import Callable

import neps.space.neps_spaces.parameters
from neps.space.neps_spaces import neps_space


def generate_possible_config_strings(
    pipeline: neps.space.neps_spaces.parameters.Pipeline,
    resolved_pipeline_attr_getter: Callable[
        [neps.space.neps_spaces.parameters.Pipeline],
        neps.space.neps_spaces.parameters.Operation,
    ],
    num_resolutions: int = 50_000,
):
    result = set()

    for _ in range(num_resolutions):
        resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)
        attr = resolved_pipeline_attr_getter(resolved_pipeline)
        config_string = neps_space.convert_operation_to_string(attr)
        result.add(config_string)

    return result
