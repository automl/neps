"""Tests that `optimizer_info.yaml` fully describes the optimizer, whatever form it is
given in, and that runs can be resumed from it.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import neps
from neps import algorithms
from neps.exceptions import NePSError
from neps.optimizers import OptimizerInfo, load_optimizer
from neps.space import HPOFloat, HPOInteger, SearchSpace
from neps.state import BudgetInfo, NePSState, OptimizationState, SeedSnapshot


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        {"x": HPOFloat(0, 1), "epochs": HPOInteger(1, 27, is_fidelity=True)}
    )


def _create_or_load(path: Path, info: OptimizerInfo) -> NePSState:
    return NePSState.create_or_load(
        path=path,
        optimizer_info=info,
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
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
    _, from_partial = load_optimizer(partial(algorithms.hyperband, eta=4), space)  # type: ignore
    _, from_string = load_optimizer(("hyperband", {"eta": 4}), space)
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
        _, info = load_optimizer(lambda s: _fake_optimizer(s), space)  # type: ignore
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
    # Older versions only recorded the keywords of a `partial`
    root = tmp_path / "run"
    _create_or_load(root, OptimizerInfo(name="hyperband", info={"eta": 4}))

    _, same = load_optimizer(partial(algorithms.hyperband, eta=4), space)  # type: ignore
    _create_or_load(root, same)

    _, different = load_optimizer(partial(algorithms.hyperband, eta=5), space)  # type: ignore
    with pytest.raises(NePSError, match="optimizer info on disk does not match"):
        _create_or_load(root, different)


def test_auto_resume_of_custom_optimizer_raises_clear_error(
    tmp_path: Path, space: SearchSpace
) -> None:
    root = tmp_path / "run"
    _create_or_load(root, OptimizerInfo(name="<lambda>", info={}))

    with pytest.raises(ValueError, match="custom optimizer '<lambda>'"):
        neps.run(
            evaluate_pipeline=lambda x, epochs: x + epochs,
            pipeline_space=space,
            root_directory=root,
            evaluations_to_spend=1,
        )


def test_bracket_optimizer_records_rung_layout(
    tmp_path: Path, space: SearchSpace
) -> None:
    _, info = load_optimizer("hyperband", space)
    derived = info["derived"]
    assert derived["bracket_type"] == "hyperband"
    assert derived["fidelity"] == {"name": "epochs", "lower": 1, "upper": 27}
    assert derived["rung_to_fidelity"] == {0: 1, 1: 3, 2: 9, 3: 27}
    assert len(derived["bracket_layouts"]) == 4

    root = tmp_path / "run"
    _create_or_load(root, info)
    assert neps.load_optimizer_info(root) == info
    _create_or_load(root, info)  # `derived` is not compared when resuming
