from __future__ import annotations

import pytest

from neps.space.neps_spaces import neps_space, string_formatter
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
            _PARAMS.resample(),
            _CONST.resample(),
            _VAR.resample(),
        ),
    )

    _UNARY = Operation(
        operator="Unary",
        args=(
            _UNARY_FUN.resample(),
            _POINTER.resample(),
        ),
    )

    _BINARY = Operation(
        operator="Binary",
        args=(
            _BINARY_FUN.resample(),
            _POINTER.resample(),
            _POINTER.resample(),
        ),
    )

    _TERNARY = Operation(
        operator="Ternary",
        args=(
            _TERNARY_FUN.resample(),
            _POINTER.resample(),
            _POINTER.resample(),
            _POINTER.resample(),
        ),
    )

    _F_ARGS = Categorical(
        choices=(
            _UNARY.resample(),
            _BINARY.resample(),
            _TERNARY.resample(),
        ),
    )

    _F = Operation(
        operator="Function",
        args=(_F_ARGS.resample(),),
        kwargs={"var": _VAR.resample()},
    )

    _L_ARGS = Categorical(
        choices=(
            (_F.resample(),),
            (
                _F.resample(),
                Resampled("_L"),
            ),
        ),
    )

    _L = Operation(
        operator="Line_operator",
        args=_L_ARGS.resample(),
    )

    P = Operation(
        operator="Program",
        args=(_L.resample(),),
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
    pretty_config = string_formatter.format_value(p)
    assert pretty_config
