from __future__ import annotations

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import config_string, neps_space
from neps.space.neps_spaces.parameters import (
    Categorical,
    Operation,
    PipelineSpace,
    Resampled,
)


class GrammarLike(PipelineSpace):
    _id = Operation(operator="Identity")
    _three = Operation(operator="Conv2D-3")
    _one = Operation(operator="Conv2D-1")
    _reluconvbn = Operation(operator="ReLUConvBN")

    _O = Categorical(choices=(_three, _one, _id))

    _C0 = Operation(
        operator="Sequential",
        args=(Resampled(_O),),
    )
    _C1 = Operation(
        operator="Sequential",
        args=(
            Resampled(_O),
            Resampled("S"),
            _reluconvbn,
        ),
    )
    _C2 = Operation(
        operator="Sequential",
        args=(
            Resampled(_O),
            Resampled("S"),
        ),
    )
    _C3 = Operation(
        operator="Sequential",
        args=(Resampled("S"),),
    )
    _C = Categorical(
        choices=(
            Resampled(_C0),
            Resampled(_C1),
            Resampled(_C2),
            Resampled(_C3),
        ),
    )

    _S0 = Operation(
        operator="Sequential",
        args=(Resampled(_C),),
    )
    _S1 = Operation(
        operator="Sequential",
        args=(_reluconvbn,),
    )
    _S2 = Operation(
        operator="Sequential",
        args=(Resampled("S"),),
    )
    _S3 = Operation(
        operator="Sequential",
        args=(
            Resampled("S"),
            Resampled(_C),
        ),
    )
    _S4 = Operation(
        operator="Sequential",
        args=(
            Resampled(_O),
            Resampled(_O),
            Resampled(_O),
        ),
    )
    _S5 = Operation(
        operator="Sequential",
        args=(
            Resampled("S"),
            Resampled("S"),
            Resampled(_O),
            Resampled(_O),
            Resampled(_O),
            Resampled(_O),
            Resampled(_O),
            Resampled(_O),
        ),
    )
    S = Categorical(
        choices=(
            Resampled(_S0),
            Resampled(_S1),
            Resampled(_S2),
            Resampled(_S3),
            Resampled(_S4),
            Resampled(_S5),
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
            (Resampled(_O),),
            (
                Resampled(_O),
                Resampled("S"),
                _reluconvbn,
            ),
            (
                Resampled(_O),
                Resampled("S"),
            ),
            (Resampled("S"),),
        ),
    )
    _C = Operation(
        operator="Sequential",
        args=Resampled(_C_ARGS),
    )

    _S_ARGS = Categorical(
        choices=(
            (Resampled(_C),),
            (_reluconvbn,),
            (Resampled("S"),),
            (
                Resampled("S"),
                Resampled(_C),
            ),
            (
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
            ),
            (
                Resampled("S"),
                Resampled("S"),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
            ),
        ),
    )
    S = Operation(
        operator="Sequential",
        args=Resampled(_S_ARGS),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = GrammarLike()

    try:
        resolved_pipeline, _ = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")

    s = resolved_pipeline.S
    s_config_string = neps_space.convert_operation_to_string(s)
    assert s_config_string
    pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    assert pretty_config


@pytest.mark.repeat(500)
def test_resolve_alt():
    pipeline = GrammarLikeAlt()

    try:
        resolved_pipeline, _ = neps_space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")

    s = resolved_pipeline.S
    s_config_string = neps_space.convert_operation_to_string(s)
    assert s_config_string
    pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    assert pretty_config


def test_resolve_context():
    samplings_to_make = {
        "Resolvable.S::categorical__6": 5,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__4": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__4": (
            3
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            4
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[4].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[5].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__4": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[3].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[5].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.sampled_value.resampled_operation.args[7].resampled_categorical::categorical__3": (
            1
        ),
    }
    expected_s_config_string = (
        "(Sequential (Sequential (Sequential (ReLUConvBN)) (Sequential (Conv2D-3)"
        " (Sequential (Sequential (Sequential (Sequential (Identity) (Conv2D-3)"
        " (Identity)))) (Sequential (ReLUConvBN)) (Conv2D-3) (Identity) (Conv2D-1)"
        " (Conv2D-3) (Conv2D-1) (Identity)) (ReLUConvBN))) (Sequential (Sequential"
        " (Sequential (Sequential (Identity) (Sequential (ReLUConvBN)))))) (Conv2D-1)"
        " (Conv2D-1) (Identity) (Identity) (Conv2D-1) (Conv2D-1))"
    )

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
    s_config_string = neps_space.convert_operation_to_string(s)
    assert s_config_string
    assert s_config_string == expected_s_config_string


def test_resolve_context_alt():
    samplings_to_make = {
        "Resolvable.S.args.resampled_categorical::categorical__6": 3,
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__4": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[3].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[4].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[6].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__4": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[3].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[6].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[7].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            5
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__6": (
            3
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__6": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical::categorical__4": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[2].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[4].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[7].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[3].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[4].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[5].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[6].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.S.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[1].resampled_operation.args.resampled_categorical.sampled_value.collection[7].resampled_categorical::categorical__3": (
            0
        ),
    }
    expected_s_config_string = (
        "(Sequential (Sequential (Sequential (Sequential (Sequential "
        "(Sequential (Conv2D-3) (Sequential (ReLUConvBN)))) (Sequential "
        "(ReLUConvBN)) (Identity) (Conv2D-3) (Conv2D-3) (Conv2D-1) (Conv2D-3) "
        "(Identity)))) (Sequential (Conv2D-3) (Sequential (Sequential "
        "(Sequential (Sequential (Sequential (ReLUConvBN)) (Sequential "
        "(Conv2D-1))) (Sequential (Sequential (ReLUConvBN))) (Conv2D-1) "
        "(Conv2D-1) (Identity) (Conv2D-1) (Conv2D-3) (Conv2D-3))) "
        "(Sequential (Sequential (ReLUConvBN)) (Sequential (Sequential "
        "(Sequential (Identity))) (Sequential (Conv2D-3))) (Conv2D-1) "
        "(Identity) (Conv2D-1) (Conv2D-1) (Conv2D-1) (Identity)) (Identity) "
        "(Identity) (Identity) (Conv2D-1) (Conv2D-1) (Conv2D-3)) (ReLUConvBN)))"
    )

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
    s_config_string = neps_space.convert_operation_to_string(s)
    assert s_config_string
    assert s_config_string == expected_s_config_string
