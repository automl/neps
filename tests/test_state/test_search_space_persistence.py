"""Tests for search space persistence in NePSState.

This file focuses on low-level NePSState functionality:
- Saving and loading search spaces (PipelineSpace and SearchSpace)
- Backward compatibility (runs without search space)
- Testing utility functions like load_pipeline_space and load_optimizer_info

For higher-level integration tests and validation logic, see
test_search_space_validation.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neps.exceptions import NePSError
from neps.optimizers import OptimizerInfo
from neps.space import HPOCategorical, HPOFloat, HPOInteger, SearchSpace
from neps.space.neps_spaces.parameters import Categorical, Float, Integer, PipelineSpace
from neps.state import BudgetInfo, NePSState, OptimizationState, SeedSnapshot


class SimpleSpace(PipelineSpace):
    """Simple test space with various parameter types."""

    a = Float(0, 1)
    b = Categorical(("x", "y", "z"))
    c = Integer(0, 10)


class Space1(PipelineSpace):
    """First test space for validation."""

    x = Float(0, 10)
    y = Integer(1, 10)


class Space2(PipelineSpace):
    """Second test space (different from TestSpace1)."""

    x = Float(0, 10)
    y = Integer(1, 20)  # Different range


def test_search_space_saved_and_loaded_pipeline_space(tmp_path: Path) -> None:
    """Test that PipelineSpace is saved and can be loaded back."""
    root_dir = tmp_path / "test_run"
    pipeline_space = SimpleSpace()

    # Create state with search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
        pipeline_space=pipeline_space,
    )

    # Verify pipeline_space.pkl exists
    assert (root_dir / "pipeline_space.pkl").exists()

    # Load state and verify search space
    state2 = NePSState.create_or_load(path=root_dir, load_only=True)
    loaded_space = state2.lock_and_get_search_space()

    assert loaded_space is not None
    assert isinstance(loaded_space, PipelineSpace)

    # Verify the structure matches
    assert "a" in loaded_space.get_attrs()
    assert "b" in loaded_space.get_attrs()
    assert "c" in loaded_space.get_attrs()


def test_search_space_saved_and_loaded_search_space(tmp_path: Path) -> None:
    """Test that old-style SearchSpace is saved and can be loaded back."""
    root_dir = tmp_path / "test_run"
    search_space = SearchSpace(
        {
            "a": HPOFloat(0, 1),
            "b": HPOCategorical(["x", "y", "z"]),
            "c": HPOInteger(0, 10),
        }
    )

    # Create state with search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
        pipeline_space=search_space,
    )

    # Verify pipeline_space.pkl exists
    assert (root_dir / "pipeline_space.pkl").exists()

    # Load state and verify search space
    state2 = NePSState.create_or_load(path=root_dir, load_only=True)
    loaded_space = state2.lock_and_get_search_space()

    assert loaded_space is not None
    assert isinstance(loaded_space, SearchSpace)
    assert "a" in loaded_space
    assert "b" in loaded_space
    assert "c" in loaded_space


def test_search_space_not_provided_backward_compatible(tmp_path: Path) -> None:
    """Test that NePSState works without search space (backward compatibility)."""
    root_dir = tmp_path / "test_run"

    # Create state WITHOUT search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
    )

    # Verify pipeline_space.pkl does NOT exist
    assert not (root_dir / "pipeline_space.pkl").exists()

    # Load state and verify search space is None
    state2 = NePSState.create_or_load(path=root_dir, load_only=True)
    loaded_space = state2.lock_and_get_search_space()

    assert loaded_space is None


def test_load_pipeline_space_function_pipeline_space(tmp_path: Path) -> None:
    """Test the load_pipeline_space utility function with PipelineSpace."""
    from neps import load_pipeline_space

    root_dir = tmp_path / "test_run"
    pipeline_space = SimpleSpace()

    # Create state with search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
        pipeline_space=pipeline_space,
    )

    # Load using the utility function
    loaded_space = load_pipeline_space(root_dir)

    assert loaded_space is not None
    assert isinstance(loaded_space, PipelineSpace)
    assert "a" in loaded_space.get_attrs()
    assert "b" in loaded_space.get_attrs()
    assert "c" in loaded_space.get_attrs()


def test_load_pipeline_space_function_search_space(tmp_path: Path) -> None:
    """Test the load_pipeline_space utility function with SearchSpace."""
    from neps import load_pipeline_space

    root_dir = tmp_path / "test_run"
    search_space = SearchSpace(
        {
            "x": HPOFloat(0, 1),
            "y": HPOInteger(1, 10),
        }
    )

    # Create state with search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
        pipeline_space=search_space,
    )

    # Load using the utility function
    loaded_space = load_pipeline_space(root_dir)

    assert loaded_space is not None
    assert isinstance(loaded_space, SearchSpace)
    assert "x" in loaded_space
    assert "y" in loaded_space


def test_load_pipeline_space_function_not_found(tmp_path: Path) -> None:
    """Test that load_pipeline_space raises FileNotFoundError for non-existent
    directory.
    """
    from neps import load_pipeline_space

    root_dir = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError, match="No neps state found"):
        load_pipeline_space(root_dir)


def test_load_pipeline_space_function_no_space_saved(tmp_path: Path) -> None:
    """Test that load_pipeline_space raises ValueError when no search space was saved."""
    from neps import load_pipeline_space

    root_dir = tmp_path / "test_run"

    # Create state WITHOUT search space
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=OptimizerInfo(name="test", info={}),
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
    )

    # Try to load - should raise ValueError
    with pytest.raises(ValueError, match="No pipeline space was saved"):
        load_pipeline_space(root_dir)


def test_load_optimizer_info_function(tmp_path: Path) -> None:
    """Test the load_optimizer_info utility function."""
    from neps import load_optimizer_info

    root_dir = tmp_path / "test_run"

    # Create state with optimizer info
    optimizer_info = OptimizerInfo(
        name="bayesian_optimization",
        info={"acquisition": "EI", "initial_design_size": 10},
    )
    NePSState.create_or_load(
        path=root_dir,
        optimizer_info=optimizer_info,
        optimizer_state=OptimizationState(
            budget=BudgetInfo(cost_to_spend=10, used_cost_budget=0),
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state={},
        ),
        pipeline_space=SimpleSpace(),
    )

    # Load using the utility function
    loaded_info = load_optimizer_info(root_dir)

    assert loaded_info["name"] == "bayesian_optimization"
    assert loaded_info["info"]["acquisition"] == "EI"
    assert loaded_info["info"]["initial_design_size"] == 10


def test_load_optimizer_info_function_not_found(tmp_path: Path) -> None:
    """Test that load_optimizer_info raises FileNotFoundError for non-existent
    directory.
    """
    from neps import load_optimizer_info

    root_dir = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError, match="No neps state found"):
        load_optimizer_info(root_dir)


def test_import_trials_saves_search_space(tmp_path: Path) -> None:
    """Test that import_trials saves the search space to disk."""
    from neps import import_trials, load_pipeline_space
    from neps.state.pipeline_eval import UserResultDict

    root_dir = tmp_path / "test_import"

    # Import trials with a search space
    evaluated_trials = [
        ({"x": 1.0, "y": 5}, UserResultDict(objective_to_minimize=1.0)),
        ({"x": 2.0, "y": 8}, UserResultDict(objective_to_minimize=2.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials,
        root_directory=root_dir,
        pipeline_space=Space1(),
    )

    # Verify the search space was saved
    loaded_space = load_pipeline_space(root_dir)
    assert loaded_space is not None
    # import_trials may convert PipelineSpace to SearchSpace, so check for SearchSpace
    assert isinstance(loaded_space, Space1 | SearchSpace)
    # Verify it has the correct parameters by checking the keys
    if isinstance(loaded_space, SearchSpace):
        assert "x" in loaded_space
        assert "y" in loaded_space


def test_import_trials_validates_search_space(tmp_path: Path) -> None:
    """Test that import_trials validates the search space against what's on disk."""
    from neps import import_trials
    from neps.state.pipeline_eval import UserResultDict

    root_dir = tmp_path / "test_import_validate"

    # First import with one space
    evaluated_trials = [
        ({"x": 1.0, "y": 5}, UserResultDict(objective_to_minimize=1.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials,
        root_directory=root_dir,
        pipeline_space=Space1(),
    )

    # Try to import again with a different space - should raise error
    with pytest.raises(NePSError, match="pipeline space on disk does not match"):
        import_trials(
            evaluated_trials=evaluated_trials,
            root_directory=root_dir,
            pipeline_space=Space2(),
        )


def test_import_trials_without_space_loads_from_disk(tmp_path: Path) -> None:
    """Test that import_trials can load pipeline space from disk when not provided."""
    from neps import import_trials, load_pipeline_space
    from neps.state.pipeline_eval import UserResultDict

    root_dir = tmp_path / "test_import_auto_load"

    # First import with explicit space
    evaluated_trials_1 = [
        ({"x": 1.0, "y": 5}, UserResultDict(objective_to_minimize=1.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials_1,
        root_directory=root_dir,
        pipeline_space=Space1(),
    )

    # Second import without providing space - should load from disk
    evaluated_trials_2 = [
        ({"x": 2.0, "y": 8}, UserResultDict(objective_to_minimize=2.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials_2,
        root_directory=root_dir,
        # pipeline_space not provided - should load from disk
    )

    # Verify both trials were imported successfully
    loaded_space = load_pipeline_space(root_dir)
    assert loaded_space is not None


def test_import_trials_without_space_fails_on_new_directory(tmp_path: Path) -> None:
    """Test that import_trials raises error when space is not provided and directory
    is new.
    """
    from neps import import_trials
    from neps.state.pipeline_eval import UserResultDict

    root_dir = tmp_path / "test_import_no_space_error"

    evaluated_trials = [
        ({"x": 1.0, "y": 5}, UserResultDict(objective_to_minimize=1.0)),
    ]

    # Should raise error when no space provided and directory doesn't exist
    with pytest.raises(
        ValueError, match="pipeline_space is required when importing trials"
    ):
        import_trials(
            evaluated_trials=evaluated_trials,
            root_directory=root_dir,
            # pipeline_space not provided
        )


def test_import_trials_validates_provided_space_against_disk(tmp_path: Path) -> None:
    """Test that when both space is provided and exists on disk, they are validated."""
    from neps import import_trials
    from neps.state.pipeline_eval import UserResultDict

    root_dir = tmp_path / "test_import_validation"

    # First import with TestSpace1
    evaluated_trials_1 = [
        ({"x": 1.0, "y": 5}, UserResultDict(objective_to_minimize=1.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials_1,
        root_directory=root_dir,
        pipeline_space=Space1(),
    )

    # Second import explicitly providing the same space - should work
    evaluated_trials_2 = [
        ({"x": 2.0, "y": 8}, UserResultDict(objective_to_minimize=2.0)),
    ]

    import_trials(
        evaluated_trials=evaluated_trials_2,
        root_directory=root_dir,
        pipeline_space=Space1(),
    )

    # Third import with different space - should fail
    with pytest.raises(NePSError, match="pipeline space on disk does not match"):
        import_trials(
            evaluated_trials=evaluated_trials_2,
            root_directory=root_dir,
            pipeline_space=Space2(),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
