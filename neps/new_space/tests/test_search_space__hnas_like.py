import pytest

from neps.new_space import space


class HNASLikePipeline(space.Pipeline):
    """
    Based on the `hierarchical+shared` variant (cell block is shared everywhere).
    Across _CONVBLOCK items, _ACT and _CONV also shared. Only the _NORM changes.

    Additionally, this variant now has a PReLU operation with a float hyperparameter (init).
    The same value of that hyperparameter would is used everywhere a _PRELU is used.
    """

    # ------------------------------------------------------
    # Adding `PReLU` with a float hyperparameter `init`
    # Note that the sampled `_prelu_init_value` will be shared across all `_PRELU` uses,
    #  since no `Resampled` was requested for it
    _prelu_init_value = space.Float(
        min_value=0.1,
        max_value=0.9,
        log=False,
        prior=0.25,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    _PRELU = space.Operation(
        operator="ACT prelu",
        kwargs={"init": _prelu_init_value},
    )
    # ------------------------------------------------------

    # Added `_PRELU` to the possible `_ACT` choices
    _ACT = space.Categorical(
        choices=(
            space.Operation(operator="ACT relu"),
            space.Operation(operator="ACT hardswish"),
            space.Operation(operator="ACT mish"),
            _PRELU,
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    _CONV = space.Categorical(
        choices=(
            space.Operation(operator="CONV conv1x1"),
            space.Operation(operator="CONV conv3x3"),
            space.Operation(operator="CONV dconv3x3"),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    _NORM = space.Categorical(
        choices=(
            space.Operation(operator="NORM batch"),
            space.Operation(operator="NORM instance"),
            space.Operation(operator="NORM layer"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    _CONVBLOCK = space.Operation(
        operator="CONVBLOCK Sequential3",
        args=(
            _ACT,
            _CONV,
            space.Resampled(_NORM),
        ),
    )
    _CONVBLOCK_FULL = space.Operation(
        operator="OPS Sequential1",
        args=(space.Resampled(_CONVBLOCK),),
    )
    _OP = space.Categorical(
        choices=(
            space.Operation(operator="OPS zero"),
            space.Operation(operator="OPS id"),
            space.Operation(operator="OPS avg_pool"),
            space.Resampled(_CONVBLOCK_FULL),
        ),
        prior_index=2,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    CL = space.Operation(
        operator="CELL Cell",
        args=(
            space.Resampled(_OP),
            space.Resampled(_OP),
            space.Resampled(_OP),
            space.Resampled(_OP),
            space.Resampled(_OP),
            space.Resampled(_OP),
        ),
    )

    _C = space.Categorical(
        choices=(
            space.Operation(operator="C Sequential2", args=(CL, CL)),
            space.Operation(operator="C Sequential3", args=(CL, CL, CL)),
            space.Operation(operator="C Residual2", args=(CL, CL, CL)),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    _RESBLOCK = space.Operation(operator="resBlock")
    _DOWN = space.Categorical(
        choices=(
            space.Operation(operator="DOWN Sequential2", args=(CL, _RESBLOCK)),
            space.Operation(operator="DOWN Sequential3", args=(CL, CL, _RESBLOCK)),
            space.Operation(operator="DOWN Residual2", args=(CL, _RESBLOCK, _RESBLOCK)),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    _D0 = space.Categorical(
        choices=(
            space.Operation(
                operator="D0 Sequential3",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    CL,
                ),
            ),
            space.Operation(
                operator="D0 Sequential4",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    space.Resampled(_C),
                    CL,
                ),
            ),
            space.Operation(
                operator="D0 Residual3",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    CL,
                    CL,
                ),
            ),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    _D1 = space.Categorical(
        choices=(
            space.Operation(
                operator="D1 Sequential3",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    space.Resampled(_DOWN),
                ),
            ),
            space.Operation(
                operator="D1 Sequential4",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    space.Resampled(_C),
                    space.Resampled(_DOWN),
                ),
            ),
            space.Operation(
                operator="D1 Residual3",
                args=(
                    space.Resampled(_C),
                    space.Resampled(_C),
                    space.Resampled(_DOWN),
                    space.Resampled(_DOWN),
                ),
            ),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    _D2 = space.Categorical(
        choices=(
            space.Operation(
                operator="D2 Sequential3",
                args=(
                    space.Resampled(_D1),
                    space.Resampled(_D1),
                    space.Resampled(_D0),
                ),
            ),
            space.Operation(
                operator="D2 Sequential3",
                args=(
                    space.Resampled(_D0),
                    space.Resampled(_D1),
                    space.Resampled(_D1),
                ),
            ),
            space.Operation(
                operator="D2 Sequential4",
                args=(
                    space.Resampled(_D1),
                    space.Resampled(_D1),
                    space.Resampled(_D0),
                    space.Resampled(_D0),
                ),
            ),
        ),
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    ARCH: space.Operation = _D2


@pytest.mark.repeat(500)
def test_hnas_like():
    pipeline = HNASLikePipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert sampled_values is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == ("CL", "ARCH")


@pytest.mark.repeat(500)
def test_hnas_like_string():
    pipeline = HNASLikePipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    arch = resolved_pipeline.ARCH
    arch_config_string = space.to_config_string(arch)
    assert arch_config_string
    pretty_config = space.config_string.ConfigString(arch_config_string).pretty_format()
    assert pretty_config

    cl = resolved_pipeline.CL
    cl_config_string = space.to_config_string(cl)
    assert cl_config_string
    pretty_config = space.config_string.ConfigString(cl_config_string).pretty_format()
    assert pretty_config


# todo: add code in the context so that after resolving, the unused sampling_values are removed
# todo: add code in the sampling name so that an ID of the Domain is written instead of just `sampled_categorical`.
#  This would be useful to identify if after a mutation, if a predefined choice can still be used.
#  Even better, write a description of the domain, enough to know if boundaries match


def test_hnas_like_context():
    samplings_to_make = {
        "CL.args[0].resampled_categorical.sampled_categorical": 3,
        "CL.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_operation.args[0].sampled_categorical": 0,
        "CL.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_operation.args[1].sampled_categorical": 2,
        "CL.args[0].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_operation.args[2].resampled_categorical.sampled_categorical": 0,
        "CL.args[1].resampled_categorical.sampled_categorical": 0,
        "CL.args[2].resampled_categorical.sampled_categorical": 1,
        "CL.args[3].resampled_categorical.sampled_categorical": 2,
        "CL.args[4].resampled_categorical.sampled_categorical": 3,
        "CL.args[4].resampled_categorical.sampled_value.resampled_operation.args[0].resampled_operation.args[2].resampled_categorical.sampled_categorical": 2,
        "CL.args[5].resampled_categorical.sampled_categorical": 0,
        "ARCH.sampled_categorical": 1,
        "ARCH.sampled_value.args[0].resampled_categorical.sampled_categorical": 2,
        "ARCH.sampled_value.args[0].resampled_categorical.sampled_value.args[0].resampled_categorical.sampled_categorical": 2,
        "ARCH.sampled_value.args[0].resampled_categorical.sampled_value.args[1].resampled_categorical.sampled_categorical": 0,
        "ARCH.sampled_value.args[1].resampled_categorical.sampled_categorical": 2,
        "ARCH.sampled_value.args[1].resampled_categorical.sampled_value.args[0].resampled_categorical.sampled_categorical": 0,
        "ARCH.sampled_value.args[1].resampled_categorical.sampled_value.args[1].resampled_categorical.sampled_categorical": 0,
        "ARCH.sampled_value.args[1].resampled_categorical.sampled_value.args[2].resampled_categorical.sampled_categorical": 0,
        "ARCH.sampled_value.args[1].resampled_categorical.sampled_value.args[3].resampled_categorical.sampled_categorical": 1,
        "ARCH.sampled_value.args[2].resampled_categorical.sampled_categorical": 2,
    }

    expected_cl_config_string = "(CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))"
    expected_arch_config_string = "(D2 Sequential3 (D0 Residual3 (C Residual2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (C Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (D1 Residual3 (C Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (C Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (DOWN Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock) (DOWN Sequential3 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock)) (D1 Residual3 (C Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (C Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (DOWN Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock) (DOWN Sequential3 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock)))"

    pipeline = HNASLikePipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline, samplings_to_make)
    assert resolved_pipeline is not None
    assert sampled_values is not None
    assert sampled_values is not samplings_to_make

    # we should have made exactly those samplings
    assert sampled_values == samplings_to_make

    cl = resolved_pipeline.CL
    cl_config_string = space.to_config_string(cl)
    assert cl_config_string
    assert cl_config_string == expected_cl_config_string
    assert "NORM batch" in cl_config_string
    assert "NORM layer" in cl_config_string

    arch = resolved_pipeline.ARCH
    arch_config_string = space.to_config_string(arch)
    assert arch_config_string
    assert arch_config_string == expected_arch_config_string
    assert cl_config_string in arch_config_string

    # print()
    # print("Sampled CELL: " + cl_config_string)
    # print("Sampled ARCH: " + arch_config_string)
    # print("Sampled values:")
    # import pprint
    #
    # # pprint.pp(sampled_values, indent=2, compact=True)
    #
    # print()
    #
    # print("ARCH received:")
    # pretty_config = space.config_string.ConfigString(arch_config_string).pretty_format()
    # print(pretty_config)
    #
    # print("Arch expected:")
    # pretty_config = space.config_string.ConfigString(expected_arch_config_string).pretty_format()
    # print(pretty_config)
