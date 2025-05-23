# import nosbench
# from nosbench.program import Program, Instruction, Pointer
# from nosbench.function import Function

import pytest

from neps.space.new_space import space
from neps.space.new_space import config_string


class nosBench(space.Pipeline):
    _UNARY_FUN = space.Categorical(
        choices=(
            space.Operation(operator="Square"),
            space.Operation(operator="Exp"),
            space.Operation(operator="Log"),
        )
    )

    _BINARY_FUN = space.Categorical(
        choices=(
            space.Operation(operator="Add"),
            space.Operation(operator="Sub"),
            space.Operation(operator="Mul"),
        )
    )

    _TERNARY_FUN = space.Categorical(
        choices=(
            space.Operation(operator="Interpolate"),
            space.Operation(operator="Bias_Correct"),
        )
    )

    _PARAMS = space.Categorical(
        choices=(
            space.Operation(operator="Params"),
            space.Operation(operator="Gradient"),
            space.Operation(operator="Opt_Step"),
        )
    )
    _CONST = space.Integer(3, 8)
    _VAR = space.Integer(9, 19)

    _POINTER = space.Categorical(
        choices=(
            space.Resampled(_PARAMS),
            space.Resampled(_CONST),
            space.Resampled(_VAR),
        ),
    )

    _UNARY = space.Operation(
        operator="Unary",
        args=(
            space.Resampled(_UNARY_FUN),
            space.Resampled(_POINTER),
        ),
    )

    _BINARY = space.Operation(
        operator="Binary",
        args=(
            space.Resampled(_BINARY_FUN),
            space.Resampled(_POINTER),
            space.Resampled(_POINTER),
        ),
    )

    _TERNARY = space.Operation(
        operator="Ternary",
        args=(
            space.Resampled(_TERNARY_FUN),
            space.Resampled(_POINTER),
            space.Resampled(_POINTER),
            space.Resampled(_POINTER),
        ),
    )

    _F_ARGS = space.Categorical(
        choices=(
            space.Resampled(_UNARY),
            space.Resampled(_BINARY),
            space.Resampled(_TERNARY),
        ),
    )

    _F = space.Operation(
        operator="Function",
        args=(space.Resampled(_F_ARGS),),
        kwargs={"var": space.Resampled(_VAR)},
    )

    _L_ARGS = space.Categorical(
        choices=(
            (space.Resampled(_F),),
            (space.Resampled(_F), space.Resampled("_L")),
        ),
    )

    _L = space.Operation(
        operator="Line_operator",
        args=space.Resampled(_L_ARGS),
    )

    P = space.Operation(
        operator="Program",
        args=(space.Resampled(_L),),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = nosBench()

    try:
        resolved_pipeline, resolution_context = space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")
        raise

    p = resolved_pipeline.P
    p_config_string = space.convert_operation_to_string(p)
    assert p_config_string
    pretty_config = config_string.ConfigString(p_config_string).pretty_format()
    assert pretty_config

    print()
    print("Config string:")
    print(pretty_config)

    # print()
    # print("Samplings made:")
    # import pprint
    #
    # pprint.pp(resolution_context.samplings_made, indent=2)
