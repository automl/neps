from __future__ import annotations

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space, string_formatter
from neps.space.neps_spaces.parameters import (
    Resample,
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
        args=(Resample(_O),),
    )
    _C1 = Operation(
        operator="Sequential",
        args=(
            Resample(_O),
            Resample("S"),
            _reluconvbn,
        ),
    )
    _C2 = Operation(
        operator="Sequential",
        args=(
            Resample(_O),
            Resample("S"),
        ),
    )
    _C3 = Operation(
        operator="Sequential",
        args=(Resample("S"),),
    )
    _C = Categorical(
        choices=(
            Resample(_C0),
            Resample(_C1),
            Resample(_C2),
            Resample(_C3),
        ),
    )

    _S0 = Operation(
        operator="Sequential",
        args=(Resample(_C),),
    )
    _S1 = Operation(
        operator="Sequential",
        args=(_reluconvbn,),
    )
    _S2 = Operation(
        operator="Sequential",
        args=(Resample("S"),),
    )
    _S3 = Operation(
        operator="Sequential",
        args=(
            Resample("S"),
            Resample(_C),
        ),
    )
    _S4 = Operation(
        operator="Sequential",
        args=(
            Resample(_O),
            Resample(_O),
            Resample(_O),
        ),
    )
    _S5 = Operation(
        operator="Sequential",
        args=(
            Resample("S"),
            Resample("S"),
            Resample(_O),
            Resample(_O),
            Resample(_O),
            Resample(_O),
            Resample(_O),
            Resample(_O),
        ),
    )
    S = Categorical(
        choices=(
            Resample(_S0),
            Resample(_S1),
            Resample(_S2),
            Resample(_S3),
            Resample(_S4),
            Resample(_S5),
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
            (Resample(_O),),
            (
                Resample(_O),
                Resample("S"),
                _reluconvbn,
            ),
            (
                Resample(_O),
                Resample("S"),
            ),
            (Resample("S"),),
        ),
    )
    _C = Operation(
        operator="Sequential",
        args=Resample(_C_ARGS),
    )

    _S_ARGS = Categorical(
        choices=(
            (Resample(_C),),
            (_reluconvbn,),
            (Resample("S"),),
            (
                Resample("S"),
                Resample(_C),
            ),
            (
                Resample(_O),
                Resample(_O),
                Resample(_O),
            ),
            (
                Resample("S"),
                Resample("S"),
                Resample(_O),
                Resample(_O),
                Resample(_O),
                Resample(_O),
                Resample(_O),
                Resample(_O),
            ),
        ),
    )
    S = Operation(
        operator="Sequential",
        args=Resample(_S_ARGS),
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
