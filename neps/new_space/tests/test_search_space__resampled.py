import pytest

from neps.new_space import space


class ActPipelineSimpleFloat(space.Pipeline):
    prelu_init_value: float = space.Float(
        min_value=0,
        max_value=1000000,
        log=False,
        prior=0.25,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    prelu_shared1 = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    prelu_shared2 = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )

    prelu_own_clone1 = space.Operation(
        operator="prelu",
        kwargs={"init": space.Resampled(prelu_init_value)},
    )
    prelu_own_clone2 = space.Operation(
        operator="prelu",
        kwargs={"init": space.Resampled(prelu_init_value)},
    )

    _prelu_init_resampled = space.Resampled(prelu_init_value)
    prelu_common_clone1 = space.Operation(
        operator="prelu",
        kwargs={"init": _prelu_init_resampled},
    )
    prelu_common_clone2 = space.Operation(
        operator="prelu",
        kwargs={"init": _prelu_init_resampled},
    )


class ActPipelineComplexInteger(space.Pipeline):
    prelu_init_value: float = space.Integer(
        min_value=0,
        max_value=1000000,
        prior=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    prelu_shared1 = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )
    prelu_shared2 = space.Operation(
        operator="prelu",
        kwargs={"init": prelu_init_value},
    )

    prelu_own_clone1 = space.Operation(
        operator="prelu",
        kwargs={"init": space.Resampled(prelu_init_value)},
    )
    prelu_own_clone2 = space.Operation(
        operator="prelu",
        kwargs={"init": space.Resampled(prelu_init_value)},
    )

    _prelu_init_resampled = space.Resampled(prelu_init_value)
    prelu_common_clone1 = space.Operation(
        operator="prelu",
        kwargs={"init": _prelu_init_resampled},
    )
    prelu_common_clone2 = space.Operation(
        operator="prelu",
        kwargs={"init": _prelu_init_resampled},
    )

    act: space.Operation = space.Operation(
        operator="sequential6",
        args=(
            prelu_shared1,
            prelu_shared2,
            prelu_own_clone1,
            prelu_own_clone2,
            prelu_common_clone1,
            prelu_common_clone2,
        ),
        kwargs={
            "prelu_shared": prelu_shared1,
            "prelu_own_clone": prelu_own_clone1,
            "prelu_common_clone": prelu_common_clone1,
            "resampled_hp_value": space.Resampled(prelu_init_value),
        },
    )


class CellPipelineCategorical(space.Pipeline):
    conv_block = space.Categorical(
        choices=(
            space.Operation(operator="conv1"),
            space.Operation(operator="conv2"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    op1 = space.Categorical(
        choices=(
            conv_block,
            space.Operation("op1"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )
    op2 = space.Categorical(
        choices=(
            space.Resampled(conv_block),
            space.Operation("op2"),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.LOW,
    )

    _resampled_op1 = space.Resampled(op1)
    cell = space.Operation(
        operator="cell",
        args=(
            op1,
            op2,
            _resampled_op1,
            space.Resampled(op2),
            _resampled_op1,
            space.Resampled(op2),
        ),
    )


@pytest.mark.repeat(200)
def test_resampled_float():
    pipeline = ActPipelineSimpleFloat()

    resolved_pipeline, _sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "prelu_init_value",
        "prelu_shared1",
        "prelu_shared2",
        "prelu_own_clone1",
        "prelu_own_clone2",
        "prelu_common_clone1",
        "prelu_common_clone2",
    )

    prelu_init_value = resolved_pipeline.prelu_init_value
    prelu_shared1 = resolved_pipeline.prelu_shared1.kwargs["init"]
    prelu_shared2 = resolved_pipeline.prelu_shared2.kwargs["init"]
    resampled_values = (
        resolved_pipeline.prelu_own_clone1.kwargs["init"],
        resolved_pipeline.prelu_own_clone2.kwargs["init"],
        resolved_pipeline.prelu_common_clone1.kwargs["init"],
        resolved_pipeline.prelu_common_clone2.kwargs["init"],
    )

    assert isinstance(prelu_init_value, float)
    assert isinstance(prelu_shared1, float)
    assert isinstance(prelu_shared2, float)
    assert all(isinstance(resampled_value, float) for resampled_value in resampled_values)

    assert prelu_init_value == prelu_shared1
    assert prelu_init_value == prelu_shared2

    assert len(set(resampled_values)) == len(resampled_values)
    assert all(resampled_value != prelu_init_value for resampled_value in resampled_values)


@pytest.mark.repeat(200)
def test_resampled_integer():
    pipeline = ActPipelineComplexInteger()

    resolved_pipeline, _sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "prelu_init_value",
        "prelu_shared1",
        "prelu_shared2",
        "prelu_own_clone1",
        "prelu_own_clone2",
        "prelu_common_clone1",
        "prelu_common_clone2",
        "act",
    )

    prelu_init_value = resolved_pipeline.prelu_init_value
    prelu_shared1 = resolved_pipeline.prelu_shared1.kwargs["init"]
    prelu_shared2 = resolved_pipeline.prelu_shared2.kwargs["init"]
    resampled_values = (
        resolved_pipeline.prelu_own_clone1.kwargs["init"],
        resolved_pipeline.prelu_own_clone2.kwargs["init"],
        resolved_pipeline.prelu_common_clone1.kwargs["init"],
        resolved_pipeline.prelu_common_clone2.kwargs["init"],
    )

    assert isinstance(prelu_init_value, int)
    assert isinstance(prelu_shared1, int)
    assert isinstance(prelu_shared2, int)
    assert all(isinstance(resampled_value, int) for resampled_value in resampled_values)

    assert prelu_init_value == prelu_shared1
    assert prelu_init_value == prelu_shared2

    assert len(set(resampled_values)) == len(resampled_values)
    assert all(resampled_value != prelu_init_value for resampled_value in resampled_values)

    act = resolved_pipeline.act

    act_args = tuple(op.kwargs["init"] for op in act.args)
    sampled_values = (prelu_shared1, prelu_shared2, *resampled_values)
    assert len(act_args) == len(sampled_values)
    for (act_arg, sampled_value) in zip(act_args, sampled_values):
        assert act_arg is sampled_value

    act_resampled_prelu_shared = act.kwargs["prelu_shared"].kwargs["init"]
    act_resampled_prelu_own_clone = act.kwargs["prelu_own_clone"].kwargs["init"]
    act_resampled_prelu_common_clone = act.kwargs["prelu_common_clone"].kwargs["init"]

    assert isinstance(act_resampled_prelu_shared, int)
    assert isinstance(act_resampled_prelu_own_clone, int)
    assert isinstance(act_resampled_prelu_common_clone, int)

    assert act_resampled_prelu_shared == prelu_init_value
    assert act_resampled_prelu_own_clone != prelu_init_value
    assert act_resampled_prelu_common_clone != prelu_init_value
    assert act_resampled_prelu_own_clone != act_resampled_prelu_common_clone

    act_resampled_hp_value = act.kwargs["resampled_hp_value"]
    assert isinstance(act_resampled_hp_value, int)
    assert act_resampled_hp_value != prelu_init_value
    assert all(resampled_value != act_resampled_hp_value for resampled_value in resampled_values)


@pytest.mark.repeat(200)
def test_resampled_categorical():
    pipeline = CellPipelineCategorical()

    resolved_pipeline, _sampled_values = space.resolve(pipeline)
    assert resolved_pipeline is not None
    assert tuple(resolved_pipeline.get_attrs().keys()) == (
        "conv_block",
        "op1",
        "op2",
        "cell",
    )

    conv_block = resolved_pipeline.conv_block
    assert conv_block is not pipeline.conv_block

    op1 = resolved_pipeline.op1
    op2 = resolved_pipeline.op2
    assert op1 is not pipeline.op1
    assert op2 is not pipeline.op2

    assert isinstance(op1, space.Operation)
    assert isinstance(op2, space.Operation)

    assert (op1 is conv_block) or (op1.operator == "op1")
    assert op2.operator == "conv1" or op2.operator == "conv2" or op2.operator == "op2"

    cell = resolved_pipeline.cell
    assert cell is not pipeline.cell

    cell_args1 = cell.args[0]
    cell_args2 = cell.args[1]
    cell_args3 = cell.args[2]
    cell_args4 = cell.args[3]
    cell_args5 = cell.args[4]
    cell_args6 = cell.args[5]

    assert cell_args1 is op1
    assert cell_args2 is op2
    # todo: think about what more tests we can have for cell_args[3-6]
