from __future__ import annotations

import pytest

from neps.space.neps_spaces import config_string, neps_space
from neps.space.neps_spaces.parameters import (
    Categorical,
    Integer,
    Operation,
    PipelineSpace,
    Resampled,
)


class NosBench(PipelineSpace):
    _UNARY_FUN = Categorical(
        choices=(
            Operation(operator="Square"),
            Operation(operator="Exp"),
            Operation(operator="Log"),
        )
    )

    _BINARY_FUN = Categorical(
        choices=(
            Operation(operator="Add"),
            Operation(operator="Sub"),
            Operation(operator="Mul"),
        )
    )

    _TERNARY_FUN = Categorical(
        choices=(
            Operation(operator="Interpolate"),
            Operation(operator="Bias_Correct"),
        )
    )

    _PARAMS = Categorical(
        choices=(
            Operation(operator="Params"),
            Operation(operator="Gradient"),
            Operation(operator="Opt_Step"),
        )
    )
    _CONST = Integer(3, 8)
    _VAR = Integer(9, 19)

    _POINTER = Categorical(
        choices=(
            Resampled(_PARAMS),
            Resampled(_CONST),
            Resampled(_VAR),
        ),
    )

    _UNARY = Operation(
        operator="Unary",
        args=(
            Resampled(_UNARY_FUN),
            Resampled(_POINTER),
        ),
    )

    _BINARY = Operation(
        operator="Binary",
        args=(
            Resampled(_BINARY_FUN),
            Resampled(_POINTER),
            Resampled(_POINTER),
        ),
    )

    _TERNARY = Operation(
        operator="Ternary",
        args=(
            Resampled(_TERNARY_FUN),
            Resampled(_POINTER),
            Resampled(_POINTER),
            Resampled(_POINTER),
        ),
    )

    _F_ARGS = Categorical(
        choices=(
            Resampled(_UNARY),
            Resampled(_BINARY),
            Resampled(_TERNARY),
        ),
    )

    _F = Operation(
        operator="Function",
        args=(Resampled(_F_ARGS),),
        kwargs={"var": Resampled(_VAR)},
    )

    _L_ARGS = Categorical(
        choices=(
            (Resampled(_F),),
            (
                Resampled(_F),
                Resampled("_L"),
            ),
        ),
    )

    _L = Operation(
        operator="Line_operator",
        args=Resampled(_L_ARGS),
    )

    P = Operation(
        operator="Program",
        args=(Resampled(_L),),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = NosBench()

    try:
        resolved_pipeline, _ = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")
        raise

    p = resolved_pipeline.P
    p_config_string = neps_space.convert_operation_to_string(p)
    assert p_config_string
    pretty_config = config_string.ConfigString(p_config_string).pretty_format()
    assert pretty_config
