"""Simplified tests for basic AutoScAn functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import autoscan
from autoscan.optimizers import algorithms
from autoscan.space.autoscan_spaces.parameters import (
    Float,
    Integer,
    PipelineSpace,
)


class SimpleSpace(PipelineSpace):
    """Simple space for testing."""

    x = Float(lower=0.0, upper=1.0)
    y = Integer(lower=1, upper=10)


def simple_evaluation(x: float, y: int) -> float:
    """Simple evaluation function."""
    return x + y


def test_basic_autoscan_run():
    """Test that basic AutoScAn run functionality works."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "basic_test"

        # Run optimization
        autoscan.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.autoscan_random_search,
            root_directory=str(root_directory),
            worker_evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check that optimization ran and created some files
        assert root_directory.exists()

        # Should have created some evaluation files
        files = list(root_directory.rglob("*"))
        assert len(files) > 0, "Should have created some files"


def test_autoscan_optimization_with_dict_return():
    """Test AutoScAn optimization with evaluation function returning dict."""

    def dict_evaluation(x: float, y: int) -> dict:
        return {
            "objective_to_minimize": x + y,
            "additional_metric": x * y,
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "dict_test"

        # Run optimization
        autoscan.run(
            evaluate_pipeline=dict_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.autoscan_random_search,
            root_directory=str(root_directory),
            worker_evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check that optimization completed
        assert root_directory.exists()


def test_different_autoscan_optimizers():
    """Test that different AutoScAn optimizers work."""
    optimizers_to_test = [
        algorithms.autoscan_random_search,
        algorithms.complex_random_search,
    ]

    for optimizer in optimizers_to_test:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_directory = Path(tmp_dir) / f"optimizer_{optimizer.__name__}"

            # Run optimization
            autoscan.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=SimpleSpace(),
                optimizer=optimizer,
                root_directory=str(root_directory),
                worker_evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Check that optimization completed
            assert root_directory.exists()


def test_autoscan_status_functionality():
    """Test that autoscan.status works after optimization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "status_test"

        # Run optimization
        autoscan.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.autoscan_random_search,
            root_directory=str(root_directory),
            worker_evaluations_to_spend=5,
            overwrite_root_directory=True,
        )

        # Test status functionality (should not raise an error)
        try:
            autoscan.status(str(root_directory))
        except (FileNotFoundError, ValueError, KeyError) as e:
            pytest.fail(f"autoscan.status should work after optimization: {e}")


def test_evaluation_results_are_recorded():
    """Test that evaluation results are properly recorded."""
    # Track evaluations
    evaluations_called = []

    def tracking_evaluation(x: float, y: int) -> float:
        result = x + y
        evaluations_called.append((x, y, result))
        return result

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "tracking_test"

        # Run optimization
        autoscan.run(
            evaluate_pipeline=tracking_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.autoscan_random_search,
            root_directory=str(root_directory),
            worker_evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check that evaluations were called
        assert len(evaluations_called) == 3, (
            f"Expected 3 evaluations, got {len(evaluations_called)}"
        )

        # Check that all results are reasonable
        for x, y, result in evaluations_called:
            assert 0.0 <= x <= 1.0, f"x should be in [0,1], got {x}"
            assert 1 <= y <= 10, f"y should be in [1,10], got {y}"
            assert result == x + y, f"Result should be x+y, got {result} != {x}+{y}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
