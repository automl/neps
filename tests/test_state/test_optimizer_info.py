"""Tests that `optimizer_info.yaml` fully describes the optimizer, whatever form it is
given in, and that runs can be resumed from it.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import extrapolaw
from extrapolaw import algorithms
from extrapolaw.exceptions import ExtrapolawError
from extrapolaw.optimizers import OptimizerInfo, load_optimizer
from extrapolaw.space import HPOFloat, HPOInteger, SearchSpace
from extrapolaw.space.extrapolaw_spaces.parameters import (
    Float,
    Integer,
    PipelineSpace,
)
from extrapolaw.state import BudgetInfo, ExtrapolawState, OptimizationState, SeedSnapshot


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace({"x": HPOFloat(0, 1), "y": HPOInteger(1, 27)})


def _create_or_load(path: Path, info: OptimizerInfo) -> ExtrapolawState:
    return ExtrapolawState.create_or_load(
        path=path,
        optimizer_info=info,
        optimizer_state=OptimizationState(
            budget=BudgetInfo(worker_cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
    )


def _ask(*_args: Any, **_kwargs: Any) -> None:
    return None


def _fake_optimizer(space: Any, a: int = 1, b: tuple = (1, 2), **kwargs: Any) -> Any:
    return _ask


def _fake_optimizer_space_second(a: int, space: Any, b: int = 2) -> Any:
    return _ask


class _CallableOptimizer:
    def __call__(self, space: Any, a: int = 1) -> Any:  # noqa: ARG002
        return _ask


def test_partial_records_defaults(space: SearchSpace) -> None:
    _, from_partial = load_optimizer(
        partial(algorithms.bayesian_optimization, initial_design_size=4),  # type: ignore
        space,
    )
    _, from_string = load_optimizer(
        ("bayesian_optimization", {"initial_design_size": 4}), space
    )
    assert from_partial == from_string


def test_partial_positional_and_var_keyword_args_are_recorded(
    space: SearchSpace,
) -> None:
    _, info = load_optimizer(partial(_fake_optimizer_space_second, 10), space)  # type: ignore
    assert info == {"name": "_fake_optimizer_space_second", "info": {"a": 10, "b": 2}}

    _, info = load_optimizer(partial(_fake_optimizer, extra=5), space)  # type: ignore
    assert info["info"] == {"a": 1, "b": [1, 2], "extra": 5}


def test_callable_instance_is_named_after_its_class(space: SearchSpace) -> None:
    _, info = load_optimizer(_CallableOptimizer(), space)  # type: ignore
    assert info == {"name": "_CallableOptimizer", "info": {"a": 1}}


def test_lambda_warns(space: SearchSpace, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, info = load_optimizer(  # type: ignore
            lambda s: _fake_optimizer(s),  # noqa: PLW0108
            space,
        )
    assert info["name"] == "<lambda>"
    assert "lambda" in caplog.text


def test_non_yaml_values_do_not_break_resume(tmp_path: Path, space: SearchSpace) -> None:
    _, info = load_optimizer(partial(_fake_optimizer, extra=len), space)  # type: ignore
    assert info["info"]["extra"] == "builtins.len"

    root = tmp_path / "run"
    _create_or_load(root, info)
    _create_or_load(root, info)  # Tuples and functions compare equal after YAML


def test_resume_accepts_settings_not_recorded_on_disk(
    tmp_path: Path, space: SearchSpace
) -> None:
    root = tmp_path / "run"
    _create_or_load(
        root,
        OptimizerInfo(name="bayesian_optimization", info={"initial_design_size": 4}),
    )

    _, same = load_optimizer(
        partial(algorithms.bayesian_optimization, initial_design_size=4),  # type: ignore
        space,
    )
    _create_or_load(root, same)

    _, different = load_optimizer(
        partial(algorithms.bayesian_optimization, initial_design_size=5),  # type: ignore
        space,
    )
    with pytest.raises(ExtrapolawError, match="optimizer info on disk does not match"):
        _create_or_load(root, different)


def test_auto_resume_of_custom_optimizer_raises_clear_error(tmp_path: Path) -> None:
    class _PipelineSpace(PipelineSpace):
        x = Float(0, 1)
        epochs = Integer(1, 27)

    root = tmp_path / "run"
    _create_or_load(root, OptimizerInfo(name="<lambda>", info={}))

    with pytest.raises(ValueError, match="custom optimizer '<lambda>'"):
        extrapolaw.acquire(
            evaluate_pipeline=lambda x, epochs: x + epochs,
            pipeline_space=_PipelineSpace(),
            root_directory=root,
            worker_evaluations_to_spend=1,
        )
