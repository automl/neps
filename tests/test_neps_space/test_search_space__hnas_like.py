from __future__ import annotations

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import config_string, neps_space
from neps.space.neps_spaces.parameters import (
    Categorical,
    Float,
    Operation,
    PipelineSpace,
    Resampled,
)


class HNASLikePipeline(PipelineSpace):
    """Based on the `hierarchical+shared` variant (cell block is shared everywhere).
    Across _CONVBLOCK items, _ACT and _CONV also shared. Only the _NORM changes.

    Additionally, this variant now has a PReLU operation with a float hyperparameter (init).
    The same value of that hyperparameter would is used everywhere a _PRELU is used.
    """

    # ------------------------------------------------------
    # Adding `PReLU` with a float hyperparameter `init`
    # Note that the sampled `_prelu_init_value` will be shared across all `_PRELU` uses,
    #  since no `Resampled` was requested for it
    _prelu_init_value = Float(min_value=0.1, max_value=0.9)
    _PRELU = Operation(
        operator="ACT prelu",
        kwargs={"init": _prelu_init_value},
    )
    # ------------------------------------------------------

    # Added `_PRELU` to the possible `_ACT` choices
    _ACT = Categorical(
        choices=(
            Operation(operator="ACT relu"),
            Operation(operator="ACT hardswish"),
            Operation(operator="ACT mish"),
            _PRELU,
        ),
    )
    _CONV = Categorical(
        choices=(
            Operation(operator="CONV conv1x1"),
            Operation(operator="CONV conv3x3"),
            Operation(operator="CONV dconv3x3"),
        ),
    )
    _NORM = Categorical(
        choices=(
            Operation(operator="NORM batch"),
            Operation(operator="NORM instance"),
            Operation(operator="NORM layer"),
        ),
    )

    _CONVBLOCK = Operation(
        operator="CONVBLOCK Sequential3",
        args=(
            _ACT,
            _CONV,
            Resampled(_NORM),
        ),
    )
    _CONVBLOCK_FULL = Operation(
        operator="OPS Sequential1",
        args=(Resampled(_CONVBLOCK),),
    )
    _OP = Categorical(
        choices=(
            Operation(operator="OPS zero"),
            Operation(operator="OPS id"),
            Operation(operator="OPS avg_pool"),
            Resampled(_CONVBLOCK_FULL),
        ),
    )

    CL = Operation(
        operator="CELL Cell",
        args=(
            Resampled(_OP),
            Resampled(_OP),
            Resampled(_OP),
            Resampled(_OP),
            Resampled(_OP),
            Resampled(_OP),
        ),
    )

    _C = Categorical(
        choices=(
            Operation(operator="C Sequential2", args=(CL, CL)),
            Operation(operator="C Sequential3", args=(CL, CL, CL)),
            Operation(operator="C Residual2", args=(CL, CL, CL)),
        ),
    )

    _RESBLOCK = Operation(operator="resBlock")
    _DOWN = Categorical(
        choices=(
            Operation(operator="DOWN Sequential2", args=(CL, _RESBLOCK)),
            Operation(operator="DOWN Sequential3", args=(CL, CL, _RESBLOCK)),
            Operation(operator="DOWN Residual2", args=(CL, _RESBLOCK, _RESBLOCK)),
        ),
    )

    _D0 = Categorical(
        choices=(
            Operation(
                operator="D0 Sequential3",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    CL,
                ),
            ),
            Operation(
                operator="D0 Sequential4",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    Resampled(_C),
                    CL,
                ),
            ),
            Operation(
                operator="D0 Residual3",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    CL,
                    CL,
                ),
            ),
        ),
    )
    _D1 = Categorical(
        choices=(
            Operation(
                operator="D1 Sequential3",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    Resampled(_DOWN),
                ),
            ),
            Operation(
                operator="D1 Sequential4",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    Resampled(_C),
                    Resampled(_DOWN),
                ),
            ),
            Operation(
                operator="D1 Residual3",
                args=(
                    Resampled(_C),
                    Resampled(_C),
                    Resampled(_DOWN),
                    Resampled(_DOWN),
                ),
            ),
        ),
    )

    _D2 = Categorical(
        choices=(
            Operation(
                operator="D2 Sequential3",
                args=(
                    Resampled(_D1),
                    Resampled(_D1),
                    Resampled(_D0),
                ),
            ),
            Operation(
                operator="D2 Sequential3",
                args=(
                    Resampled(_D0),
                    Resampled(_D1),
                    Resampled(_D1),
                ),
            ),
            Operation(
                operator="D2 Sequential4",
                args=(
                    Resampled(_D1),
                    Resampled(_D1),
                    Resampled(_D0),
                    Resampled(_D0),
                ),
            ),
        ),
    )

    ARCH: Operation = _D2


@pytest.mark.repeat(500)
def test_hnas_like():
    pipeline = HNASLikePipeline()

    resolved_pipeline, resolution_context = neps_space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert resolution_context.samplings_made is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == ("CL", "ARCH")


@pytest.mark.repeat(500)
def test_hnas_like_string():
    pipeline = HNASLikePipeline()

    resolved_pipeline, _ = neps_space.resolve(pipeline)

    arch = resolved_pipeline.ARCH
    arch_config_string = neps_space.convert_operation_to_string(arch)
    assert arch_config_string
    pretty_config = config_string.ConfigString(arch_config_string).pretty_format()
    assert pretty_config

    cl = resolved_pipeline.CL
    cl_config_string = neps_space.convert_operation_to_string(cl)
    assert cl_config_string
    pretty_config = config_string.ConfigString(cl_config_string).pretty_format()
    assert pretty_config


def test_hnas_like_context():
    samplings_to_make = {
        "Resolvable.CL.args.sequence[0].resampled_categorical::categorical__4": 3,
        "Resolvable.CL.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_operation.args.sequence[0]::categorical__4": (
            0
        ),
        "Resolvable.CL.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_operation.args.sequence[1]::categorical__3": (
            2
        ),
        "Resolvable.CL.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_operation.args.sequence[2].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.CL.args.sequence[1].resampled_categorical::categorical__4": 0,
        "Resolvable.CL.args.sequence[2].resampled_categorical::categorical__4": 1,
        "Resolvable.CL.args.sequence[3].resampled_categorical::categorical__4": 2,
        "Resolvable.CL.args.sequence[4].resampled_categorical::categorical__4": 3,
        "Resolvable.CL.args.sequence[4].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_operation.args.sequence[2].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.CL.args.sequence[5].resampled_categorical::categorical__4": 0,
        "Resolvable.ARCH::categorical__3": 1,
        "Resolvable.ARCH.sampled_value.args.sequence[0].resampled_categorical::categorical__3": 2,
        "Resolvable.ARCH.sampled_value.args.sequence[0].resampled_categorical.sampled_value.args.sequence[0].resampled_categorical::categorical__3": (
            2
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[0].resampled_categorical.sampled_value.args.sequence[1].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[1].resampled_categorical::categorical__3": 2,
        "Resolvable.ARCH.sampled_value.args.sequence[1].resampled_categorical.sampled_value.args.sequence[0].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[1].resampled_categorical.sampled_value.args.sequence[1].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[1].resampled_categorical.sampled_value.args.sequence[2].resampled_categorical::categorical__3": (
            0
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[1].resampled_categorical.sampled_value.args.sequence[3].resampled_categorical::categorical__3": (
            1
        ),
        "Resolvable.ARCH.sampled_value.args.sequence[2].resampled_categorical::categorical__3": 2,
    }

    expected_cl_config_string = (
        "(CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3)"
        " (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))"
    )
    expected_arch_config_string = (
        "(D2 Sequential3 (D0 Residual3 (C Residual2 (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero))) (C Sequential2 (CELL Cell (OPS"
        " Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch)))"
        " (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT"
        " relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero))) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero))) (D1 Residual3 (C Sequential2 (CELL"
        " Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM"
        " batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell"
        " (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM"
        " batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (C"
        " Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV"
        " dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))"
        " (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3)"
        " (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))) (DOWN"
        " Sequential2 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV"
        " dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero))"
        " resBlock) (DOWN Sequential3 (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3"
        " (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool)"
        " (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM"
        " layer))) (OPS zero)) (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT"
        " relu) (CONV dconv3x3) (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS"
        " Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer)))"
        " (OPS zero)) resBlock)) (D1 Residual3 (C Sequential2 (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero))) (C Sequential2 (CELL Cell (OPS"
        " Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch)))"
        " (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT"
        " relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell (OPS Sequential1"
        " (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch))) (OPS zero)"
        " (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu)"
        " (CONV dconv3x3) (NORM layer))) (OPS zero))) (DOWN Sequential2 (CELL Cell (OPS"
        " Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM batch)))"
        " (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK Sequential3 (ACT"
        " relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock) (DOWN Sequential3"
        " (CELL Cell (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3)"
        " (NORM batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) (CELL Cell"
        " (OPS Sequential1 (CONVBLOCK Sequential3 (ACT relu) (CONV dconv3x3) (NORM"
        " batch))) (OPS zero) (OPS id) (OPS avg_pool) (OPS Sequential1 (CONVBLOCK"
        " Sequential3 (ACT relu) (CONV dconv3x3) (NORM layer))) (OPS zero)) resBlock)))"
    )

    pipeline = HNASLikePipeline()

    resolved_pipeline, resolution_context = neps_space.resolve(
        pipeline=pipeline,
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

    cl = resolved_pipeline.CL
    cl_config_string = neps_space.convert_operation_to_string(cl)
    assert cl_config_string
    assert cl_config_string == expected_cl_config_string
    assert "NORM batch" in cl_config_string
    assert "NORM layer" in cl_config_string

    arch = resolved_pipeline.ARCH
    arch_config_string = neps_space.convert_operation_to_string(arch)
    assert arch_config_string
    assert arch_config_string == expected_arch_config_string
    assert cl_config_string in arch_config_string
