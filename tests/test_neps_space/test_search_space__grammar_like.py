from __future__ import annotations

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space, string_formatter
from neps.space.neps_spaces.parameters import (
    ByName,
    Categorical,
    Operation,
    PipelineSpace,
)


class GrammarLike(PipelineSpace):
    _id = Operation(operator="Identity")
    _three = Operation(operator="Conv2D-3")
    _one = Operation(operator="Conv2D-1")
    _reluconvbn = Operation(operator="ReLUConvBN")

    _O = Categorical(choices=(_three, _one, _id))

    _C0 = Operation(
        operator="Sequential",
        args=(_O.resample(),),
    )
    _C1 = Operation(
        operator="Sequential",
        args=(
            _O.resample(),
            ByName("S").resample(),
            _reluconvbn,
        ),
    )
    _C2 = Operation(
        operator="Sequential",
        args=(
            _O.resample(),
            ByName("S").resample(),
        ),
    )
    _C3 = Operation(
        operator="Sequential",
        args=(ByName("S").resample(),),
    )
    _C = Categorical(
        choices=(
            _C0.resample(),
            _C1.resample(),
            _C2.resample(),
            _C3.resample(),
        ),
    )

    _S0 = Operation(
        operator="Sequential",
        args=(_C.resample(),),
    )
    _S1 = Operation(
        operator="Sequential",
        args=(_reluconvbn,),
    )
    _S2 = Operation(
        operator="Sequential",
        args=(ByName("S").resample(),),
    )
    _S3 = Operation(
        operator="Sequential",
        args=(
            ByName("S").resample(),
            _C.resample(),
        ),
    )
    _S4 = Operation(
        operator="Sequential",
        args=(
            _O.resample(),
            _O.resample(),
            _O.resample(),
        ),
    )
    _S5 = Operation(
        operator="Sequential",
        args=(
            ByName("S").resample(),
            ByName("S").resample(),
            _O.resample(),
            _O.resample(),
            _O.resample(),
            _O.resample(),
            _O.resample(),
            _O.resample(),
        ),
    )
    S = Categorical(
        choices=(
            _S0.resample(),
            _S1.resample(),
            _S2.resample(),
            _S3.resample(),
            _S4.resample(),
            _S5.resample(),
        ),
    )


class GrammarLikeAlt(PipelineSpace):
    _id = Operation(operator="Identity")
    _three = Operation(operator="Conv2D-3")
    _one = Operation(operator="Conv2D-1")
    _reluconvbn = Operation(operator="ReLUConvBN")

    _O = Categorical(choices=(_three, _one, _id))

    _C_ARGS = Categorical(
        choices=(
            (_O.resample(),),
            (
                _O.resample(),
                ByName("S").resample(),
                _reluconvbn,
            ),
            (
                _O.resample(),
                ByName("S").resample(),
            ),
            (ByName("S").resample(),),
        ),
    )
    _C = Operation(
        operator="Sequential",
        args=_C_ARGS.resample(),
    )

    _S_ARGS = Categorical(
        choices=(
            (_C.resample(),),
            (_reluconvbn,),
            (ByName("S").resample(),),
            (
                ByName("S").resample(),
                _C.resample(),
            ),
            (
                _O.resample(),
                _O.resample(),
                _O.resample(),
            ),
            (
                ByName("S").resample(),
                ByName("S").resample(),
                _O.resample(),
                _O.resample(),
                _O.resample(),
                _O.resample(),
                _O.resample(),
                _O.resample(),
            ),
        ),
    )
    S = Operation(
        operator="Sequential",
        args=_S_ARGS.resample(),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = GrammarLike()

    try:
        resolved_pipeline, _ = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")

    s = resolved_pipeline.S
    s_config_string = string_formatter.format_value(s)
    assert s_config_string


@pytest.mark.repeat(500)
def test_resolve_alt():
    pipeline = GrammarLikeAlt()

    try:
        resolved_pipeline, _ = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")

    s = resolved_pipeline.S
    s_config_string = string_formatter.format_value(s)
    assert s_config_string


def test_resolve_context():
    samplings_to_make = {
        "Resolvable.S::categorical__6": 5,
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__4": (
            3
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            4
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[2].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[4].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[5].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__4": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[3].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[5].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args.sequence[7].resampled_categorical::categorical__3": (
            1
        ),
    }

    pipeline = GrammarLike()

    resolved_pipeline, resolution_context = neps_space.resolve(
        pipeline,
        domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
            predefined_samplings=samplings_to_make,
        ),
    )
    sampled_values = resolution_context.samplings_made

    assert resolved_pipeline is not None
    assert sampled_values is not None
    assert sampled_values is not samplings_to_make
    assert sampled_values == samplings_to_make
    assert list(sampled_values.items()) == list(samplings_to_make.items())

    # we should have made exactly those samplings
    assert sampled_values == samplings_to_make

    s = resolved_pipeline.S
    s_config_string = string_formatter.format_value(s)
    assert s_config_string
    # Verify the config contains expected operation names (format may be compact or multiline)
    assert "Sequential" in s_config_string
    assert "ReLUConvBN" in s_config_string
    assert "Conv2D-3" in s_config_string
    assert "Identity" in s_config_string
    assert "Conv2D-1" in s_config_string


def test_resolve_context_alt():
    samplings_to_make = {
        "Resolvable.S.args.resampled_categorical::categorical__6": 3,
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__4": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[3].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[4].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[6].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__4": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[3].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[6].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[7].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[4].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[1].resampled_operation.args.resampled_categorical.sampled_value.sequence[7].resampled_categorical::categorical__3": (
            0
        ),
    }

    pipeline = GrammarLikeAlt()

    resolved_pipeline, resolution_context = neps_space.resolve(
        pipeline,
        domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
            predefined_samplings=samplings_to_make,
        ),
    )
    sampled_values = resolution_context.samplings_made

    assert resolved_pipeline is not None
    assert sampled_values is not None
    assert sampled_values is not samplings_to_make
    assert sampled_values == samplings_to_make
    assert list(sampled_values.items()) == list(samplings_to_make.items())

    # we should have made exactly those samplings
    assert sampled_values == samplings_to_make

    s = resolved_pipeline.S
    s_config_string = string_formatter.format_value(s)
    assert s_config_string
    # Verify the config contains expected operation names (format may be compact or multiline)
    assert "Sequential" in s_config_string
    assert "ReLUConvBN" in s_config_string
    assert "Conv2D-1" in s_config_string
    assert "Identity" in s_config_string


def test_string_resample_raises_error():
    """Test that using Resample with a string raises a TypeError."""
    from neps.space.neps_spaces.parameters import Resample

    with pytest.raises(TypeError, match="Resample does not accept plain strings"):
        Resample("test_param")
