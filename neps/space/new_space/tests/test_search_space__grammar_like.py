import pytest

from neps.space.new_space import space
from neps.space.new_space import config_string


class GrammarLike(space.Pipeline):
    _id = space.Operation(operator="Identity")
    _three = space.Operation(operator="Conv2D-3")
    _one = space.Operation(operator="Conv2D-1")
    _reluconvbn = space.Operation(operator="ReLUConvBN")

    _O = space.Categorical(choices=(_three, _one, _id))

    _C0 = space.Operation(
        operator="Sequential",
        args=(space.Resampled(_O),),
    )
    _C1 = space.Operation(
        operator="Sequential",
        args=(space.Resampled(_O), space.Resampled("S"), _reluconvbn),
    )
    _C2 = space.Operation(
        operator="Sequential",
        args=(space.Resampled(_O), space.Resampled("S")),
    )
    _C3 = space.Operation(
        operator="Sequential",
        args=(space.Resampled("S"),),
    )
    _C = space.Categorical(
        choices=(
            space.Resampled(_C0),
            space.Resampled(_C1),
            space.Resampled(_C2),
            space.Resampled(_C3),
        ),
    )

    _S0 = space.Operation(
        operator="Sequential",
        args=(space.Resampled(_C),),
    )
    _S1 = space.Operation(
        operator="Sequential",
        args=(_reluconvbn,),
    )
    _S2 = space.Operation(
        operator="Sequential",
        args=(space.Resampled("S"),),
    )
    _S3 = space.Operation(
        operator="Sequential",
        args=(space.Resampled("S"), space.Resampled(_C)),
    )
    _S4 = space.Operation(
        operator="Sequential",
        args=(space.Resampled(_O), space.Resampled(_O), space.Resampled(_O)),
    )
    _S5 = space.Operation(
        operator="Sequential",
        args=(
            space.Resampled("S"),
            space.Resampled("S"),
            space.Resampled(_O),
            space.Resampled(_O),
            space.Resampled(_O),
            space.Resampled(_O),
            space.Resampled(_O),
            space.Resampled(_O),
        ),
    )
    S = space.Categorical(
        choices=(
            space.Resampled(_S0),
            space.Resampled(_S1),
            space.Resampled(_S2),
            space.Resampled(_S3),
            space.Resampled(_S4),
            space.Resampled(_S5),
        ),
    )


class GrammarLikeAlt(space.Pipeline):
    _id = space.Operation(operator="Identity")
    _three = space.Operation(operator="Conv2D-3")
    _one = space.Operation(operator="Conv2D-1")
    _reluconvbn = space.Operation(operator="ReLUConvBN")

    _O = space.Categorical(choices=(_three, _one, _id))

    _C_ARGS = space.Categorical(
        choices=(
            (space.Resampled(_O),),
            (space.Resampled(_O), space.Resampled("S"), _reluconvbn),
            (space.Resampled(_O), space.Resampled("S")),
            (space.Resampled("S"),),
        ),
    )
    _C = space.Operation(
        operator="Sequential",
        args=space.Resampled(_C_ARGS),
    )

    _S_ARGS = space.Categorical(
        choices=(
            (space.Resampled(_C),),
            (_reluconvbn,),
            (space.Resampled("S"),),
            (space.Resampled("S"), space.Resampled(_C)),
            (space.Resampled(_O), space.Resampled(_O), space.Resampled(_O)),
            (
                space.Resampled("S"),
                space.Resampled("S"),
                space.Resampled(_O),
                space.Resampled(_O),
                space.Resampled(_O),
                space.Resampled(_O),
                space.Resampled(_O),
                space.Resampled(_O),
            ),
        ),
    )
    S = space.Operation(
        operator="Sequential",
        args=space.Resampled(_S_ARGS),
    )


@pytest.mark.repeat(500)
def test_resolve():
    pipeline = GrammarLike()

    try:
        resolved_pipeline, resolution_context = space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")
        raise

    s = resolved_pipeline.S
    s_config_string = space.convert_operation_to_string(s)
    assert s_config_string
    pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    assert pretty_config

    # print()
    # print("Config string:")
    # print(pretty_config)
    #
    # print()
    # print("Samplings made:")
    # import pprint
    # pprint.pp(resolution_context.samplings_made, indent=2)


@pytest.mark.repeat(500)
def test_resolve_alt():
    pipeline = GrammarLikeAlt()

    try:
        resolved_pipeline, resolution_context = space.resolve(pipeline)
    except RecursionError:
        pytest.xfail("XFAIL due to too much recursion.")
        raise

    s = resolved_pipeline.S
    s_config_string = space.convert_operation_to_string(s)
    assert s_config_string
    pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    assert pretty_config

    # print()
    # print("Config string:")
    # print(pretty_config)
    #
    # print()
    # print("Samplings made:")
    # import pprint
    # pprint.pp(resolution_context.samplings_made, indent=2)


def test_resolve_context():
    samplings_to_make = {
        "Resolvable.S::categorical__6": 5,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 3,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__4": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": 5,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__4": 3,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 4,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__3": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[3].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[4].resampled_categorical::categorical__3": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[5].resampled_categorical::categorical__3": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[6].resampled_categorical::categorical__3": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[7].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__6": 0,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__4": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[1].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_categorical.sampled_value.resampled_operation.args[1].resampled_categorical::categorical__6": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[2].resampled_categorical::categorical__3": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[3].resampled_categorical::categorical__3": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[4].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[5].resampled_categorical::categorical__3": 2,
        "Resolvable.S.sampled_value.resampled_operation.args[6].resampled_categorical::categorical__3": 1,
        "Resolvable.S.sampled_value.resampled_operation.args[7].resampled_categorical::categorical__3": 1,
    }
    expected_s_config_string = "(Sequential (Sequential (Sequential (ReLUConvBN)) (Sequential (Conv2D-3) (Sequential (Sequential (Sequential (Sequential (Identity) (Conv2D-3) (Identity)))) (Sequential (ReLUConvBN)) (Conv2D-3) (Identity) (Conv2D-1) (Conv2D-3) (Conv2D-1) (Identity)) (ReLUConvBN))) (Sequential (Sequential (Sequential (Sequential (Identity) (Sequential (ReLUConvBN)))))) (Conv2D-1) (Conv2D-1) (Identity) (Identity) (Conv2D-1) (Conv2D-1))"

    pipeline = GrammarLike()

    resolved_pipeline, resolution_context = space.resolve(
        pipeline,
        domain_sampler=space.OnlyPredefinedValuesSampler(
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
    s_config_string = space.convert_operation_to_string(s)
    assert s_config_string
    assert s_config_string == expected_s_config_string

    # print()
    # print("Config string:")
    # pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    # print(pretty_config)
    #
    # print()
    # print("Samplings made:")
    # import pprint
    # pprint.pp(resolution_context.samplings_made, indent=2)


def test_resolve_context_alt():
    samplings_to_make = {
        "Resolvable.S.args.resampled_categorical::categorical__6": 3,
        "Resolvable.S.args[0].resampled_operation.args.resampled_categorical::categorical__6": 0,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__4": 1,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args[0].resampled_categorical::categorical__3": 2,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args.resampled_categorical::categorical__6": 3,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__6": 1,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[1].resampled_operation.args.resampled_categorical::categorical__4": 0,
        "Resolvable.S.args[0].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[1].resampled_operation.args[0].resampled_categorical::categorical__3": 0,
        "Resolvable.S.args[1].resampled_operation.args.resampled_categorical::categorical__4": 3,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__6": 3,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__6": 0,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[0].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__4": 0,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[0].resampled_operation.args[0].resampled_operation.args[0].resampled_categorical::categorical__3": 0,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args.resampled_categorical::categorical__4": 3,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[0].resampled_operation.args.resampled_categorical::categorical__6": 4,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[0].resampled_operation.args[0].resampled_categorical::categorical__3": 1,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_categorical::categorical__3": 2,
        "Resolvable.S.args[1].resampled_operation.args[0].resampled_operation.args[1].resampled_operation.args[0].resampled_operation.args[2].resampled_categorical::categorical__3": 0,
    }
    expected_s_config_string = "(Sequential (Sequential (Sequential (Identity) (Sequential (Sequential (ReLUConvBN)) (Sequential (Conv2D-3))) (ReLUConvBN))) (Sequential (Sequential (Sequential (Sequential (Conv2D-3))) (Sequential (Sequential (Conv2D-1) (Identity) (Conv2D-3))))))"

    pipeline = GrammarLikeAlt()

    resolved_pipeline, resolution_context = space.resolve(
        pipeline,
        domain_sampler=space.OnlyPredefinedValuesSampler(
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
    s_config_string = space.convert_operation_to_string(s)
    assert s_config_string
    assert s_config_string == expected_s_config_string

    # print()
    # print("Config string:")
    # pretty_config = config_string.ConfigString(s_config_string).pretty_format()
    # print(pretty_config)
    #
    # print()
    # print("Samplings made:")
    # import pprint
    # pprint.pp(resolution_context.samplings_made, indent=2)
