"""Tests for search space validation and error handling.

This file focuses on high-level integration tests through neps.run():
- Strict validation (errors on mismatched search spaces)
- Auto-loading from disk
- Error handling when search space is missing
- Integration with load_config, status, and DDP runtime

For low-level persistence tests, see test_search_space_persistence.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import neps
from neps.exceptions import NePSError
from neps.space import SearchSpace
from neps.space.neps_spaces.parameters import Float, Integer, PipelineSpace
from neps.state import NePSState


class TestSpace1(PipelineSpace):
    """First test pipeline space."""

    x = Float(0.0, 1.0)
    y = Integer(1, 10)


class TestSpace2(PipelineSpace):
    """Different test pipeline space."""

    a = Float(0.0, 2.0)
    b = Integer(5, 20)


def eval_fn1(**config):
    """Evaluation function for TestSpace1."""
    return config["x"] + config["y"]


def eval_fn2(**config):
    """Evaluation function for TestSpace2."""
    return config["a"] + config["b"]


def test_error_on_mismatched_search_space(tmp_path: Path):
    """Test that providing a different search space raises an error (strict
    validation).
    """
    root_dir = tmp_path / "test_error"

    # Create initial state with TestSpace1
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Try to continue with TestSpace2 - should raise NePSError
    with pytest.raises(NePSError, match="pipeline space on disk does not match"):
        neps.run(
            evaluate_pipeline=eval_fn2,
            pipeline_space=TestSpace2(),
            root_directory=str(root_dir),
            evaluations_to_spend=2,
        )


def test_success_without_search_space_when_on_disk(tmp_path: Path):
    """Test that not providing search space works when one exists on disk."""
    root_dir = tmp_path / "test_no_space"

    # Create initial state with TestSpace1
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Continue WITHOUT providing pipeline_space - should load from disk
    neps.run(
        evaluate_pipeline=eval_fn1,
        # pipeline_space not provided!
        root_directory=str(root_dir),
        evaluations_to_spend=3,  # Total evaluations wanted
    )

    # Verify we have at least 2 evaluations (continuation worked)
    df, _summary = neps.status(str(root_dir), print_summary=False)
    assert len(df) >= 2, f"Should have at least 2 evaluations, got {len(df)}"


def test_error_when_no_space_provided_and_none_on_disk(tmp_path: Path):
    """Test that not providing search space errors when none exists on disk."""
    root_dir = tmp_path / "test_no_space_error"

    # Try to run WITHOUT providing pipeline_space and with no existing run
    with pytest.raises(ValueError, match="pipeline_space is required"):
        neps.run(
            evaluate_pipeline=eval_fn1,
            # pipeline_space not provided and root_dir doesn't exist!
            root_directory=str(root_dir),
            evaluations_to_spend=1,
        )


def test_load_only_does_not_validate(tmp_path: Path, caplog):
    """Test that load_only=True does not validate search space."""
    root_dir = tmp_path / "test_load_only"

    # Create initial state
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Load with load_only - should not error or warn about validation
    with caplog.at_level(logging.WARNING):
        state = NePSState.create_or_load(
            path=root_dir,
            load_only=True,
        )
        loaded_space = state.lock_and_get_search_space()

    # Should not have validation errors/warnings since load_only=True
    assert not any(
        "pipeline space on disk" in record.message.lower() for record in caplog.records
    )

    # Should have loaded the original space
    assert loaded_space is not None


def test_load_config_with_wrong_space_raises_error(tmp_path: Path):
    """Test that load_config with wrong pipeline_space raises an error."""
    root_dir = tmp_path / "test_load_config_error"

    # Create run with TestSpace1
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Find a config file
    config_dir = root_dir / "configs"
    configs = [
        d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    assert len(configs) > 0, "Should have at least one config"

    config_path = configs[0] / "config.yaml"

    # Try to load with wrong pipeline_space - should raise error
    with pytest.raises(NePSError, match="pipeline_space provided does not match"):
        neps.load_config(config_path, pipeline_space=TestSpace2())


def test_load_config_without_space_auto_loads(tmp_path: Path):
    """Test that load_config without pipeline_space auto-loads from disk."""
    root_dir = tmp_path / "test_load_config_auto"

    # Create run
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Find a config file
    config_dir = root_dir / "configs"
    configs = [
        d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    assert len(configs) > 0, "Should have at least one config"

    config_path = configs[0] / "config.yaml"

    # Load config without providing space - should auto-load from disk
    config = neps.load_config(config_path)

    assert "x" in config, "Should have x parameter"


def test_ddp_runtime_loads_search_space(tmp_path: Path):
    """Test that DDP runtime path also loads search space correctly."""
    root_dir = tmp_path / "test_ddp"

    # Create initial state with search space
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Simulate DDP path - just load_only (DDP doesn't create state)
    state = NePSState.create_or_load(path=root_dir, load_only=True)
    loaded_space = state.lock_and_get_search_space()

    assert loaded_space is not None, "DDP should be able to load search space"
    # NePS converts PipelineSpace to SearchSpace internally
    assert isinstance(loaded_space, PipelineSpace | SearchSpace), "Should load correctly"


def test_status_without_space_works(tmp_path: Path):
    """Test that status works without explicit pipeline_space."""
    root_dir = tmp_path / "test_status_auto"

    # Create a run
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Status without pipeline_space - should work
    df, _summary = neps.status(str(root_dir), print_summary=False)
    assert len(df) > 0, "Should have results"


def test_status_handles_missing_search_space_gracefully(tmp_path: Path):
    """Test that status doesn't crash if search space can't be loaded."""
    root_dir = tmp_path / "test_status_missing"

    # Create a run
    neps.run(
        evaluate_pipeline=eval_fn1,
        pipeline_space=TestSpace1(),
        root_directory=str(root_dir),
        evaluations_to_spend=1,
    )

    # Delete the search space file
    search_space_file = root_dir / "pipeline_space.pkl"
    if search_space_file.exists():
        search_space_file.unlink()

    # Status with print_summary=False should work even without search space
    df, _summary = neps.status(str(root_dir), print_summary=False)
    assert len(df) > 0, "Should still get results"
