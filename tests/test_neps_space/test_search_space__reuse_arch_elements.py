from __future__ import annotations

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space, string_formatter
from neps.space.neps_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Float,
    Integer,
    Operation,
    PipelineSpace,
)


class ActPipelineSimple(PipelineSpace):
    prelu_with_args = Operation(
        operator="prelu_with_args",
        args=(0.1, 0.2),
    )
    prelu_with_kwargs = Operation(
        operator="prelu_with_kwargs",
        kwargs={"init": 0.1},
    )
    relu = Operation(operator="relu")

    act: Operation = Categorical(
        choices=(prelu_with_args, prelu_with_kwargs, relu),
    )


class ActPipelineComplex(PipelineSpace):
    prelu_init_value: float = Float(lower=0.1, upper=0.9)
    prelu = Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    act: Operation = Categorical(
        choices=(prelu,),
    )


class FixedPipeline(PipelineSpace):
    prelu_init_value: float = 0.5
    prelu = Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    act = prelu


_conv_choices_low = ("conv1x1", "conv3x3")
_conv_choices_high = ("conv5x5", "conv9x9")
_conv_choices_prior_confidence_choices = (
    ConfidenceLevel.LOW,
    ConfidenceLevel.MEDIUM,
    ConfidenceLevel.HIGH,
)


class ConvPipeline(PipelineSpace):
    conv_choices_prior_index: int = Integer(
        lower=0,
        upper=1,
        log=False,
        prior=0,
        prior_confidence=ConfidenceLevel.LOW,
    )
    conv_choices_prior_confidence: ConfidenceLevel = Categorical(
        choices=_conv_choices_prior_confidence_choices,
        prior=1,
        prior_confidence=ConfidenceLevel.LOW,
    )
    conv_choices: tuple[str, ...] = Categorical(
        choices=(_conv_choices_low, _conv_choices_high),
        prior=conv_choices_prior_index,
        prior_confidence=conv_choices_prior_confidence,
    )

    _conv1: str = Categorical(
        choices=conv_choices,
    )
    _conv2: str = Categorical(
        choices=conv_choices,
    )

    conv_block: Operation = Categorical(
        choices=(
            Operation(
                operator="sequential3",
                args=[_conv1, _conv2, _conv1],
            ),
        ),
    )


class CellPipeline(PipelineSpace):
    _act = Operation(operator="relu")
    _conv = Operation(operator="conv3x3")
    _norm = Operation(operator="batch")

    conv_block = Operation(operator="sequential3", args=(_act, _conv, _norm))

    op1 = Categorical(
        choices=(
            conv_block,
            Operation(operator="zero"),
            Operation(operator="avg_pool"),
        ),
    )
    op2 = Categorical(
        choices=(
            conv_block,
            Operation(operator="zero"),
            Operation(operator="avg_pool"),
        ),
    )

    _some_int = 2
    _some_float = Float(lower=0.5, upper=0.5)

    cell = Operation(
        operator="cell",
        args=(op1, op2, op1, op2, op1, op2),
        kwargs={"float_hp": _some_float, "int_hp": _some_int},
    )


@pytest.mark.repeat(50)
def test_nested_simple():
    pipeline = ActPipelineSimple()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "prelu_with_args",
        "prelu_with_kwargs",
        "relu",
        "act",
    )

    assert resolved_pipeline.prelu_with_kwargs is pipeline.prelu_with_kwargs
    assert resolved_pipeline.prelu_with_args is pipeline.prelu_with_args
    assert resolved_pipeline.relu is pipeline.relu

    assert resolved_pipeline.act in (
        resolved_pipeline.prelu_with_kwargs,
        resolved_pipeline.prelu_with_args,
        resolved_pipeline.relu,
    )


@pytest.mark.repeat(50)
def test_nested_simple_string():
    # Format is now always expanded, check for content
    pipeline = ActPipelineSimple()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = string_formatter.format_value(act)
    assert act_config_string

    # Check for one of the possible operations
    is_relu = "relu" in act_config_string.lower()
    is_prelu_args = "prelu_with_args" in act_config_string and "0.1" in act_config_string
    is_prelu_kwargs = (
        "prelu_with_kwargs" in act_config_string and "init=0.1" in act_config_string
    )
    assert is_relu or is_prelu_args or is_prelu_kwargs


@pytest.mark.repeat(50)
def test_nested_complex():
    pipeline = ActPipelineComplex()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "prelu_init_value",
        "prelu",
        "act",
    )

    prelu_init_value = resolved_pipeline.prelu_init_value
    assert 0.1 <= prelu_init_value <= 0.9

    prelu = resolved_pipeline.prelu
    assert prelu.operator == "prelu"
    assert isinstance(prelu.kwargs["init"], float)
    assert prelu.kwargs["init"] is prelu_init_value
    assert not prelu.args

    act = resolved_pipeline.act
    assert act.operator == "prelu"
    assert act is prelu


@pytest.mark.repeat(50)
def test_nested_complex_string():
    pipeline = ActPipelineComplex()

    resolved_pipeline, _ = neps_space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = string_formatter.format_value(act)
    assert act_config_string

    # Format is now expanded, check for content
    assert "prelu" in act_config_string
    assert "init=" in act_config_string
    # Extract the init value (should be between 0.1 and 0.9)
    import re

    match = re.search(r"init=([\d.]+)", act_config_string)
    assert match is not None
    init_value = float(match.group(1))
    assert 0.1 <= init_value <= 0.9


def test_fixed_pipeline():
    pipeline = FixedPipeline()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == tuple(
        pipeline.get_attrs().keys()
    )

    assert resolved_pipeline.prelu_init_value == pipeline.prelu_init_value
    assert resolved_pipeline.prelu is pipeline.prelu
    assert resolved_pipeline.act is pipeline.act
    assert resolved_pipeline is pipeline


def test_fixed_pipeline_string():
    pipeline = FixedPipeline()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = string_formatter.format_value(act)
    assert act_config_string
    # Check content rather than exact format (now always expanded)
    assert "prelu" in act_config_string
    assert "init=0.5" in act_config_string


@pytest.mark.repeat(50)
def test_simple_reuse():
    pipeline = ConvPipeline()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "conv_choices_prior_index",
        "conv_choices_prior_confidence",
        "conv_choices",
        "conv_block",
    )

    conv_choices_prior_index = resolved_pipeline.conv_choices_prior_index
    assert conv_choices_prior_index in (0, 1)

    conv_choices_prior_confidence = resolved_pipeline.conv_choices_prior_confidence
    assert conv_choices_prior_confidence in _conv_choices_prior_confidence_choices

    conv_choices = resolved_pipeline.conv_choices
    assert conv_choices in (_conv_choices_low, _conv_choices_high)

    conv_block = resolved_pipeline.conv_block
    assert conv_block.operator == "sequential3"
    for conv in conv_block.args:
        assert conv in conv_choices
    assert conv_block.args[0] == conv_block.args[2]


@pytest.mark.repeat(50)
def test_simple_reuse_string():
    # Check that the formatted string reflects the reuse pattern correctly
    # Format is now always expanded, so check semantic content
    pipeline = ConvPipeline()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    conv_block = resolved_pipeline.conv_block
    conv_block_config_string = string_formatter.format_value(conv_block)
    assert conv_block_config_string

    # Should contain sequential3 and three conv operations
    assert "sequential3" in conv_block_config_string
    assert conv_block_config_string.count("conv") == 3

    # Extract the three conv operations - they should follow the reuse pattern
    # where first and third are the same
    import re

    convs = re.findall(r"(conv\dx\d)", conv_block_config_string)
    assert len(convs) == 3
    assert convs[0] == convs[2], f"First and third conv should match: {convs}"


@pytest.mark.repeat(50)
def test_shared_complex():
    pipeline = CellPipeline()

    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    assert resolved_pipeline is not pipeline
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "conv_block",
        "op1",
        "op2",
        "cell",
    )

    conv_block = resolved_pipeline.conv_block
    assert conv_block is pipeline.conv_block

    op1 = resolved_pipeline.op1
    op2 = resolved_pipeline.op2
    assert op1 is not pipeline.op1
    assert op2 is not pipeline.op2
    assert isinstance(op1, Operation)
    assert isinstance(op2, Operation)

    if op1 is op2:
        assert op1 is conv_block
    else:
        assert op1.operator in {"zero", "avg_pool", "sequential3"}
        assert op2.operator in {"zero", "avg_pool", "sequential3"}
        if op1.operator == "sequential3" or op2.operator == "sequential3":
            assert op1.operator != op2.operator

    cell = resolved_pipeline.cell
    assert cell is not pipeline.cell
    assert cell.operator == "cell"
    assert cell.args[0] is op1
    assert cell.args[1] is op2
    assert cell.args[2] is op1
    assert cell.args[3] is op2
    assert cell.args[4] is op1
    assert cell.args[5] is op2
    assert len(cell.kwargs) == 2
    assert cell.kwargs["float_hp"] == 0.5
    assert cell.kwargs["int_hp"] == 2


@pytest.mark.repeat(50)
def test_shared_complex_string():
    # The new formatter outputs all operations in full, rather than using
    # references for shared operations. Check for key elements instead of exact format.

    pipeline = CellPipeline()
    resolved_pipeline, _resolution_context = neps_space.resolve(pipeline)

    cell = resolved_pipeline.cell
    cell_config_string = string_formatter.format_value(cell)

    # Verify essential elements are present
    assert cell_config_string
    assert cell_config_string.startswith("cell(")
    assert "float_hp=0.5" in cell_config_string
    assert "int_hp=2" in cell_config_string

    # Check that the operation types that could appear are present
    # (at least one of avg_pool, zero, or sequential3 should appear)
    has_operation = (
        "avg_pool()" in cell_config_string
        or "zero()" in cell_config_string
        or "sequential3(" in cell_config_string
    )
    assert has_operation


def test_shared_complex_context():
    # todo: move the context testing part to its own test file.
    #  This one should only do the reuse tests

    # todo: add a more complex test, where we have hidden Categorical choices.
    #  E.g. add Resample along the way

    samplings_to_make = {
        "Resolvable.op1::categorical__3": 2,
        "Resolvable.op2::categorical__3": 1,
        "Resolvable.cell.kwargs.mapping_value{float_hp}::float__0.5_0.5_False": 0.5,
    }

    pipeline = CellPipeline()

    resolved_pipeline_first, _resolution_context_first = neps_space.resolve(
        pipeline=pipeline,
        domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
            predefined_samplings=samplings_to_make,
        ),
    )
    sampled_values_first = _resolution_context_first.samplings_made

    assert resolved_pipeline_first is not pipeline
    assert sampled_values_first is not None
    assert sampled_values_first is not samplings_to_make
    assert sampled_values_first == samplings_to_make
    assert list(sampled_values_first.items()) == list(samplings_to_make.items())

    resolved_pipeline_second, _resolution_context_second = neps_space.resolve(
        pipeline=pipeline,
        domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
            predefined_samplings=samplings_to_make,
        ),
    )
    sampled_values_second = _resolution_context_second.samplings_made

    assert resolved_pipeline_second is not pipeline
    assert resolved_pipeline_second is not None
    assert sampled_values_second is not samplings_to_make
    assert sampled_values_second == samplings_to_make
    assert list(sampled_values_second.items()) == list(samplings_to_make.items())

    # the second resolution should give us a new object
    assert resolved_pipeline_second is not resolved_pipeline_first

    # The new formatter outputs operations in full rather than using references.
    # Check that both resolutions produce the same format and contain expected operations.
    config_str_first = string_formatter.format_value(resolved_pipeline_first.cell)
    config_str_second = string_formatter.format_value(resolved_pipeline_second.cell)

    # Both resolutions with same samplings should produce identical output
    assert config_str_first == config_str_second

    # Check essential elements are present
    assert config_str_first.startswith("cell(")
    assert "avg_pool()" in config_str_first
    assert "zero()" in config_str_first
    assert "float_hp=0.5" in config_str_first
    assert "int_hp=2" in config_str_first
