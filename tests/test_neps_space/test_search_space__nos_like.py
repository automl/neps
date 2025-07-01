from __future__ import annotations

import pytest

from neps.space.neps_spaces import config_string, neps_space


class NosBench(neps_space.Pipeline):
    _UNARY_FUN = neps_space.Categorical(
        choices=(
            neps_space.Operation(operator="Square"),
            neps_space.Operation(operator="Exp"),
            neps_space.Operation(operator="Log"),
        )
    )

    _BINARY_FUN = neps_space.Categorical(
        choices=(
            neps_space.Operation(operator="Add"),
            neps_space.Operation(operator="Sub"),
            neps_space.Operation(operator="Mul"),
        )
    )

    _TERNARY_FUN = neps_space.Categorical(
        choices=(
            neps_space.Operation(operator="Interpolate"),
            neps_space.Operation(operator="Bias_Correct"),
        )
    )

    _PARAMS = neps_space.Categorical(
        choices=(
            neps_space.Operation(operator="Params"),
            neps_space.Operation(operator="Gradient"),
            neps_space.Operation(operator="Opt_Step"),
        )
    )
    _CONST = neps_space.Integer(3, 8)
    _VAR = neps_space.Integer(9, 19)

    _POINTER = neps_space.Categorical(
        choices=(
            neps_space.Resampled(_PARAMS),
            neps_space.Resampled(_CONST),
            neps_space.Resampled(_VAR),
        ),
    )

    _UNARY = neps_space.Operation(
        operator="Unary",
        args=(
            neps_space.Resampled(_UNARY_FUN),
            neps_space.Resampled(_POINTER),
        ),
    )

    _BINARY = neps_space.Operation(
        operator="Binary",
        args=(
            neps_space.Resampled(_BINARY_FUN),
            neps_space.Resampled(_POINTER),
            neps_space.Resampled(_POINTER),
        ),
    )

    _TERNARY = neps_space.Operation(
        operator="Ternary",
        args=(
            neps_space.Resampled(_TERNARY_FUN),
            neps_space.Resampled(_POINTER),
            neps_space.Resampled(_POINTER),
            neps_space.Resampled(_POINTER),
        ),
    )

    _F_ARGS = neps_space.Categorical(
        choices=(
            neps_space.Resampled(_UNARY),
            neps_space.Resampled(_BINARY),
            neps_space.Resampled(_TERNARY),
        ),
    )

    _F = neps_space.Operation(
        operator="Function",
        args=(neps_space.Resampled(_F_ARGS),),
        kwargs={"var": neps_space.Resampled(_VAR)},
    )

    _L_ARGS = neps_space.Categorical(
        choices=(
            (neps_space.Resampled(_F),),
            (neps_space.Resampled(_F), neps_space.Resampled("_L")),
        ),
    )

    _L = neps_space.Operation(
        operator="Line_operator",
        args=neps_space.Resampled(_L_ARGS),
    )

    P = neps_space.Operation(
        operator="Program",
        args=(neps_space.Resampled(_L),),
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
