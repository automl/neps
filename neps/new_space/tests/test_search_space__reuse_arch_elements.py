import pytest

from neps.new_space import space


class ActPipelineSimple(space.Pipeline):
    prelu = space.Operation(
        operator="prelu",
        kwargs={"init": 0.1},
    )
    relu = space.Operation(operator="relu")

    act: space.Operation = space.Categorical(
        choices=(prelu, relu),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.HIGH,
    )


class ActPipelineComplex(space.Pipeline):
    prelu_init_value: float = space.Float(
        min_value=0.1,
        max_value=0.9,
        log=False,
        prior=0.25,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    prelu = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    act: space.Operation = space.Categorical(
        choices=(prelu,),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.HIGH,
    )


class FixedPipeline(space.Pipeline):
    prelu_init_value: float = 0.5
    prelu = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    act = prelu


_conv_choices_low = ("conv1x1", "conv3x3")
_conv_choices_high = ("conv5x5", "conv9x9")
_conv_choices_prior_confidence_choices = (
    space.ConfidenceLevel.LOW,
    space.ConfidenceLevel.MEDIUM,
    space.ConfidenceLevel.HIGH,
)


class ConvPipeline(space.Pipeline):
    conv_choices_prior_index: int = space.Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    conv_choices_prior_confidence: space.ConfidenceLevel = space.Categorical(
        choices=_conv_choices_prior_confidence_choices,
        prior_index=1,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    conv_choices: tuple[str, ...] = space.Categorical(
        choices=(_conv_choices_low, _conv_choices_high),
        prior_index=conv_choices_prior_index,
        prior_confidence=conv_choices_prior_confidence,
    )

    _conv1: str = space.Categorical(
        choices=conv_choices,
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    _conv2: str = space.Categorical(
        choices=conv_choices,
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    conv_block: space.Operation = space.Categorical(
        choices=(
            space.Operation(
                operator="sequential3",
                args=[_conv1, _conv2, _conv1],
            ),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )


class CellPipeline(space.Pipeline):
    _act = space.Operation(operator="relu")
    _conv = space.Operation(operator="conv3x3")
    _norm = space.Operation(operator="batch")

    conv_block = space.Operation(operator="sequential3", args=(_act, _conv, _norm))

    op1 = space.Categorical(
        choices=(
            conv_block,
            space.Operation(operator="zero"),
            space.Operation(operator="avg_pool"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    op2 = space.Categorical(
        choices=(
            conv_block,
            space.Operation(operator="zero"),
            space.Operation(operator="avg_pool"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    _some_int = 2
    _some_float = space.Float(
        min_value=0.5,
        max_value=0.5,
        log=False,
        prior=0.5,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    cell = space.Operation(
        operator="cell",
        args=(op1, op2, op1, op2, op1, op2),
        kwargs={"float_hp": _some_float, "int_hp": _some_int},
    )


@pytest.mark.repeat(50)
def test_nested_simple():
    pipeline = ActPipelineSimple()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == ("prelu", "relu", "act")

    assert resolved_pipeline.prelu is pipeline.prelu
    assert resolved_pipeline.relu is pipeline.relu


@pytest.mark.repeat(50)
def test_nested_simple_string():
    possible_cell_config_strings = {
        "(relu)",
        "(prelu {'init': 0.1})",
    }

    pipeline = ActPipelineSimple()

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = space.to_config_string(act)
    assert act_config_string
    assert act_config_string in possible_cell_config_strings


@pytest.mark.repeat(50)
def test_nested_complex():
    pipeline = ActPipelineComplex()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == ("prelu_init_value", "prelu", "act")

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

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = space.to_config_string(act)
    assert act_config_string

    # expected to look like: "(prelu {'init': 0.1087727907176638})"
    expected_prefix = "(prelu {'init': "
    expected_ending = "})"
    assert act_config_string.startswith(expected_prefix)
    assert act_config_string.endswith(expected_ending)
    assert 0.1 <= float(act_config_string[len(expected_prefix) : -len(expected_ending)]) <= 0.9


def test_fixed_pipeline():
    pipeline = FixedPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == tuple(pipeline.get_attrs().keys())

    assert resolved_pipeline.prelu_init_value == pipeline.prelu_init_value
    assert resolved_pipeline.prelu is pipeline.prelu
    assert resolved_pipeline.act is pipeline.act
    assert resolved_pipeline is pipeline


def test_fixed_pipeline_string():
    pipeline = FixedPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    act = resolved_pipeline.act
    act_config_string = space.to_config_string(act)
    assert act_config_string
    assert act_config_string == "(prelu {'init': 0.5})"


@pytest.mark.repeat(50)
def test_simple_reuse():
    pipeline = ConvPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "conv_choices_prior_index",
        "conv_choices_prior_confidence",
        "conv_choices",
        "conv_block",
    )

    conv_choices_prior_index = resolved_pipeline.conv_choices_prior_index
    assert conv_choices_prior_index == 0 or conv_choices_prior_index == 1

    conv_choices_prior_confidence = resolved_pipeline.conv_choices_prior_confidence
    assert conv_choices_prior_confidence in _conv_choices_prior_confidence_choices

    conv_choices = resolved_pipeline.conv_choices
    assert conv_choices == _conv_choices_low or conv_choices == _conv_choices_high

    conv_block = resolved_pipeline.conv_block
    assert conv_block.operator == "sequential3"
    for conv in conv_block.args:
        assert conv in conv_choices
    assert conv_block.args[0] == conv_block.args[2]


@pytest.mark.repeat(50)
def test_simple_reuse_string():
    possible_conv_block_config_strings = {
        "(sequential3 (conv1x1) (conv1x1) (conv1x1))",
        "(sequential3 (conv1x1) (conv3x3) (conv1x1))",
        "(sequential3 (conv3x3) (conv1x1) (conv3x3))",
        "(sequential3 (conv3x3) (conv3x3) (conv3x3))",
        "(sequential3 (conv5x5) (conv5x5) (conv5x5))",
        "(sequential3 (conv5x5) (conv9x9) (conv5x5))",
        "(sequential3 (conv9x9) (conv5x5) (conv9x9))",
        "(sequential3 (conv9x9) (conv9x9) (conv9x9))",
    }

    pipeline = ConvPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    conv_block = resolved_pipeline.conv_block
    conv_block_config_string = space.to_config_string(conv_block)
    assert conv_block_config_string
    assert conv_block_config_string in possible_conv_block_config_strings


@pytest.mark.repeat(50)
def test_shared_complex():
    pipeline = CellPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)
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
    assert isinstance(op1, space.Operation)
    assert isinstance(op2, space.Operation)

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
    possible_cell_config_strings = {
        "(cell {'float_hp': 0.5, 'int_hp': 2} (avg_pool) (avg_pool) (avg_pool) (avg_pool) (avg_pool) (avg_pool))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (zero) (sequential3 (relu) (conv3x3) (batch)) (zero) (sequential3 (relu) (conv3x3) (batch)) (zero) (sequential3 (relu) (conv3x3) (batch)))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (sequential3 (relu) (conv3x3) (batch)) (avg_pool) (sequential3 (relu) (conv3x3) (batch)) (avg_pool) (sequential3 (relu) (conv3x3) (batch)) (avg_pool))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (zero) (zero) (zero) (zero) (zero) (zero))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (zero) (avg_pool) (zero) (avg_pool) (zero) (avg_pool))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (sequential3 (relu) (conv3x3) (batch)) (sequential3 (relu) (conv3x3) (batch)) (sequential3 (relu) (conv3x3) (batch)) (sequential3 (relu) (conv3x3) (batch)) (sequential3 (relu) (conv3x3) (batch)) (sequential3 (relu) (conv3x3) (batch)))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (avg_pool) (zero) (avg_pool) (zero) (avg_pool) (zero))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (sequential3 (relu) (conv3x3) (batch)) (zero) (sequential3 (relu) (conv3x3) (batch)) (zero) (sequential3 (relu) (conv3x3) (batch)) (zero))",
        "(cell {'float_hp': 0.5, 'int_hp': 2} (avg_pool) (sequential3 (relu) (conv3x3) (batch)) (avg_pool) (sequential3 (relu) (conv3x3) (batch)) (avg_pool) (sequential3 (relu) (conv3x3) (batch)))",
    }

    pipeline = CellPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline)

    cell = resolved_pipeline.cell
    cell_config_string = space.to_config_string(cell)
    assert cell_config_string
    assert cell_config_string in possible_cell_config_strings


def test_shared_complex_context():
    # todo: move the context testing part to its own test file.
    #  This one should only do the reuse tests

    # todo: add a more complex test, where we have hidden Categorical choices.
    #  E.g. add Resampled along the way

    samplings_to_make = {
        "op1.sampled_categorical": 2,
        "op2.sampled_categorical": 1,
        "cell.kwargs__float_hp.sampled_float": 0.5,
    }

    pipeline = CellPipeline()

    resolved_pipeline, sampled_values = space.resolve(pipeline, samplings_to_make)
    assert resolved_pipeline is not pipeline
    assert sampled_values is not samplings_to_make
    assert sampled_values == samplings_to_make

    resolved_pipeline_new, sampled_values_new = space.resolve(pipeline, samplings_to_make)
    assert resolved_pipeline_new is not pipeline
    assert sampled_values_new is not samplings_to_make
    assert sampled_values_new == samplings_to_make

    # the second resolution should give us a new object
    assert resolved_pipeline_new is not resolved_pipeline

    expected_config_string: str = (
        "(cell {'float_hp': 0.5, 'int_hp': 2} (avg_pool) (zero) (avg_pool) (zero) (avg_pool) (zero))"
    )

    # however, their final results should be the same thing
    assert space.to_config_string(resolved_pipeline.cell) == expected_config_string
    assert space.to_config_string(resolved_pipeline_new.cell) == expected_config_string
