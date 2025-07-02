from __future__ import annotations

import pytest

import neps.space.neps_spaces.parameters
from neps.space.neps_spaces import config_string, neps_space


class NosBench(neps.space.neps_spaces.parameters.Pipeline):
    _UNARY_FUN = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Operation(operator="Square"),
            neps.space.neps_spaces.parameters.Operation(operator="Exp"),
            neps.space.neps_spaces.parameters.Operation(operator="Log"),
        )
    )

    _BINARY_FUN = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Operation(operator="Add"),
            neps.space.neps_spaces.parameters.Operation(operator="Sub"),
            neps.space.neps_spaces.parameters.Operation(operator="Mul"),
        )
    )

    _TERNARY_FUN = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Operation(operator="Interpolate"),
            neps.space.neps_spaces.parameters.Operation(operator="Bias_Correct"),
        )
    )

    _PARAMS = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Operation(operator="Params"),
            neps.space.neps_spaces.parameters.Operation(operator="Gradient"),
            neps.space.neps_spaces.parameters.Operation(operator="Opt_Step"),
        )
    )
    _CONST = neps.space.neps_spaces.parameters.Integer(3, 8)
    _VAR = neps.space.neps_spaces.parameters.Integer(9, 19)

    _POINTER = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Resampled(_PARAMS),
            neps.space.neps_spaces.parameters.Resampled(_CONST),
            neps.space.neps_spaces.parameters.Resampled(_VAR),
        ),
    )

    _UNARY = neps.space.neps_spaces.parameters.Operation(
        operator="Unary",
        args=(
            neps.space.neps_spaces.parameters.Resampled(_UNARY_FUN),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
        ),
    )

    _BINARY = neps.space.neps_spaces.parameters.Operation(
        operator="Binary",
        args=(
            neps.space.neps_spaces.parameters.Resampled(_BINARY_FUN),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
        ),
    )

    _TERNARY = neps.space.neps_spaces.parameters.Operation(
        operator="Ternary",
        args=(
            neps.space.neps_spaces.parameters.Resampled(_TERNARY_FUN),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
            neps.space.neps_spaces.parameters.Resampled(_POINTER),
        ),
    )

    _F_ARGS = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            neps.space.neps_spaces.parameters.Resampled(_UNARY),
            neps.space.neps_spaces.parameters.Resampled(_BINARY),
            neps.space.neps_spaces.parameters.Resampled(_TERNARY),
        ),
    )

    _F = neps.space.neps_spaces.parameters.Operation(
        operator="Function",
        args=(neps.space.neps_spaces.parameters.Resampled(_F_ARGS),),
        kwargs={"var": neps.space.neps_spaces.parameters.Resampled(_VAR)},
    )

    _L_ARGS = neps.space.neps_spaces.parameters.Categorical(
        choices=(
            (neps.space.neps_spaces.parameters.Resampled(_F),),
            (
                neps.space.neps_spaces.parameters.Resampled(_F),
                neps.space.neps_spaces.parameters.Resampled("_L"),
            ),
        ),
    )

    _L = neps.space.neps_spaces.parameters.Operation(
        operator="Line_operator",
        args=neps.space.neps_spaces.parameters.Resampled(_L_ARGS),
    )

    P = neps.space.neps_spaces.parameters.Operation(
        operator="Program",
        args=(neps.space.neps_spaces.parameters.Resampled(_L),),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = NosBench()

    try:
        resolved_pipeline, resolution_context = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")
        raise

    p = resolved_pipeline.P
    p_config_string = neps_space.convert_operation_to_string(p)
    assert p_config_string
    pretty_config = config_string.ConfigString(p_config_string).pretty_format()
    assert pretty_config
