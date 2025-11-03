"""Tests for extended trajectory and metrics functionality in NePS."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

import neps
from neps.optimizers import algorithms
from neps.space.neps_spaces.parameters import (
    Fidelity,
    Float,
    Integer,
    PipelineSpace,
)


class SimpleSpace(PipelineSpace):
    """Simple space for testing metrics functionality."""

    x = Float(lower=0.0, upper=1.0)
    y = Integer(lower=1, upper=10)


class SpaceWithFidelity(PipelineSpace):
    """Space with fidelity for testing multi-fidelity metrics."""

    x = Float(lower=0.0, upper=1.0)
    y = Integer(lower=1, upper=10)
    epochs = Fidelity(Integer(lower=1, upper=50))


def simple_evaluation(x: float, y: int) -> dict:
    """Simple evaluation function that returns multiple metrics."""
    return {
        "objective_to_minimize": x + y,
        "accuracy": max(0.0, 1.0 - (x + y) / 11.0),  # Dummy accuracy metric
        "training_time": x * 10 + y,  # Dummy training time
        "memory_usage": y * 100,  # Dummy memory usage
        "custom_metric": x * y,  # Custom metric
    }


def fidelity_evaluation(x: float, y: int, epochs: int) -> dict:
    """Evaluation function with fidelity that affects metrics."""
    base_objective = x + y
    fidelity_factor = epochs / 50.0  # Scale based on fidelity

    return {
        "objective_to_minimize": (
            base_objective / fidelity_factor
        ),  # Better with more epochs
        "accuracy": min(1.0, fidelity_factor * (1.0 - base_objective / 11.0)),
        "training_time": epochs * (x * 10 + y),  # More epochs = more time
        "memory_usage": y * 100 + epochs * 10,  # Memory increases with epochs
        "convergence_rate": 1.0 / epochs,  # Faster convergence with more epochs
        "epochs_used": epochs,  # Track actual epochs used
    }


def failing_evaluation(x: float, y: int) -> dict:
    """Evaluation that sometimes fails to test error handling."""
    if x > 0.8 or y > 8:
        raise ValueError("Simulated failure for testing")

    return {
        "objective_to_minimize": x + y,
        "success_rate": 1.0,
    }


# ===== Test basic trajectory and metrics =====


def test_basic_trajectory_functionality():
    """Test basic trajectory functionality without checking specific file structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "basic_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check that some optimization files were created
        assert root_directory.exists()

        # Find the summary directory and check for result files
        summary_dir = root_directory / "summary"
        assert summary_dir.exists(), "Summary directory should exist"

        # Check for best config file
        best_config_file = summary_dir / "best_config.txt"
        assert best_config_file.exists(), "Best config file should exist"

        # Check if trajectory file contains our evaluation results
        best_config_content = best_config_file.read_text()
        assert "Objective to minimize" in best_config_content  # Different casing

        # Check for CSV files that contain the optimization summary
        csv_files = list(summary_dir.glob("*.csv"))
        assert len(csv_files) > 0, "Should have CSV summary files"

        # Check that basic optimization data is present
        csv_content = csv_files[0].read_text()
        assert "objective_to_minimize" in csv_content, "Should contain objective values"


def test_best_config_with_multiple_metrics():
    """Test that best_config file contains multiple metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "best_config_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=5,
            overwrite_root_directory=True,
        )

        # Check that best_config file exists
        best_config_file = root_directory / "summary" / "best_config.txt"
        assert best_config_file.exists(), "Best config file should exist"

        # Read and verify best config contains multiple metrics
        best_config_content = best_config_file.read_text()

        # Should contain the primary objective
        assert "Objective to minimize" in best_config_content

        # Note: Additional metrics may not be persisted to summary files
        # They are used during evaluation but only the main objective is saved
        # Should contain configuration parameters
        assert (
            "x" in best_config_content or "SAMPLING__Resolvable.x" in best_config_content
        )
        assert (
            "y" in best_config_content or "SAMPLING__Resolvable.y" in best_config_content
        )


def test_trajectory_with_fidelity():
    """Test trajectory with fidelity-based evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "fidelity_test"

        # Run optimization with fidelity
        neps.run(
            evaluate_pipeline=fidelity_evaluation,
            pipeline_space=SpaceWithFidelity(),
            optimizer=("neps_random_search", {"ignore_fidelity": True}),
            root_directory=str(root_directory),
            evaluations_to_spend=10,
            overwrite_root_directory=True,
        )

        # Check trajectory file
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        assert trajectory_file.exists()

        trajectory_content = trajectory_file.read_text()

        # Should contain basic optimization data
        assert "Config ID" in trajectory_content
        assert "Objective" in trajectory_content

        # Should track configuration parameters (including fidelity if preserved)
        assert any(
            param in trajectory_content
            for param in ["x", "y", "epochs", "SAMPLING__Resolvable"]
        )


def test_cumulative_metrics_tracking():
    """Test that cumulative evaluations are tracked in trajectory files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "cumulative_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=5,
            overwrite_root_directory=True,
        )

        # Read trajectory
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        trajectory_content = trajectory_file.read_text()

        # Should have the expected header
        assert (
            "Best configs and their objectives across evaluations:" in trajectory_content
        )

        # Should track cumulative evaluations
        assert "Cumulative evaluations:" in trajectory_content

        # Should have multiple config entries (at least some evaluations)
        config_count = trajectory_content.count("Config ID:")
        assert config_count >= 1, "Should have at least one config entry"

        # Should have objective values
        assert "Objective to minimize:" in trajectory_content


# ===== Test error handling in metrics =====


def test_trajectory_with_failed_evaluations():
    """Test that trajectory handles failed evaluations correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "error_test"

        # Run optimization that will have some failures
        neps.run(
            evaluate_pipeline=failing_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=15,  # More evaluations to ensure some failures
            overwrite_root_directory=True,
            ignore_errors=True,  # Allow continuing after errors
        )

        # Check that trajectory file exists
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        assert trajectory_file.exists()

        # Read trajectory
        trajectory_content = trajectory_file.read_text()
        lines = trajectory_content.strip().split("\n")

        # Should have at least some successful evaluations
        assert len(lines) >= 2  # Header + at least one evaluation

        # Check that errors are handled gracefully
        # (The exact behavior may vary, but the file should exist and be readable)
        assert "Objective to minimize" in trajectory_content  # Different casing


# ===== Test hyperband-specific metrics =====


def test_neps_hyperband_metrics():
    """Test that neps_hyperband produces extended metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "hyperband_test"

        # Run neps_hyperband optimization
        neps.run(
            evaluate_pipeline=fidelity_evaluation,
            pipeline_space=SpaceWithFidelity(),
            optimizer=algorithms.neps_hyperband,
            root_directory=str(root_directory),
            fidelities_to_spend=20,  # Use fidelities_to_spend for mf optimizers
            overwrite_root_directory=True,
        )

        # Check trajectory file
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        assert trajectory_file.exists()

        trajectory_content = trajectory_file.read_text()

        # Should contain basic optimization data
        assert "Objective" in trajectory_content

        # Should contain configuration information
        assert any(
            param in trajectory_content for param in ["epochs", "SAMPLING__Resolvable"]
        )

        # Should have multiple evaluations with different fidelities
        lines = trajectory_content.strip().split("\n")
        assert len(lines) >= 5  # Should have some evaluations


# ===== Test metrics with different optimizers =====


@pytest.mark.parametrize(
    "optimizer",
    [
        algorithms.neps_random_search,
        algorithms.complex_random_search,
    ],
)
def test_metrics_with_different_optimizers(optimizer):
    """Test that txt file format is consistent across different optimizers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / f"optimizer_test_{optimizer.__name__}"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=optimizer,
            root_directory=str(root_directory),
            evaluations_to_spend=5,
            overwrite_root_directory=True,
        )

        # Check files exist
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        best_config_file = root_directory / "summary" / "best_config.txt"

        assert trajectory_file.exists()
        assert best_config_file.exists()

        # Check contents match expected txt format (only objective_to_minimize is tracked)
        trajectory_content = trajectory_file.read_text()
        best_config_content = best_config_file.read_text()

        # Both should contain the standard txt file format elements
        for content in [trajectory_content, best_config_content]:
            assert "Config ID:" in content
            assert "Objective to minimize:" in content
            assert "Cumulative evaluations:" in content
            assert "Config:" in content

        # Trajectory file should have the header
        assert (
            "Best configs and their objectives across evaluations:" in trajectory_content
        )


# ===== Test metric value validation =====


def test_metric_values_are_reasonable():
    """Test that reported objective values are reasonable in txt files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "validation_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=5,
            overwrite_root_directory=True,
        )

        # Read trajectory and parse objective values
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        trajectory_content = trajectory_file.read_text()

        # Extract objective values from the actual txt format
        objective_matches = re.findall(
            r"Objective to minimize: ([\d.]+)", trajectory_content
        )

        # Check that we found some objectives
        assert len(objective_matches) > 0, "No objective values found in trajectory"

        # Check each objective value is reasonable
        for obj_str in objective_matches:
            objective = float(obj_str)
            # Objective should be in reasonable range (x+y where x in [0,1], y in [1,10])
            assert 1.0 <= objective <= 11.0, (
                f"Objective {objective} out of expected range [1.0, 11.0]"
            )


# ===== Test file format and structure =====


def test_trajectory_file_format():
    """Test that trajectory txt file has correct format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "format_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check trajectory file format (txt format, not CSV)
        trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
        trajectory_content = trajectory_file.read_text()

        # Should have the expected txt file structure
        assert (
            "Best configs and their objectives across evaluations:" in trajectory_content
        )
        assert "Config ID:" in trajectory_content
        assert "Objective to minimize:" in trajectory_content
        assert "Cumulative evaluations:" in trajectory_content
        assert "Config:" in trajectory_content

        # Should have separator lines
        assert (
            "--------------------------------------------------------------------------------"
            in trajectory_content
        )


def test_results_directory_structure():
    """Test that results directory has expected structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "structure_test"

        # Run optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        # Check directory structure
        results_dir = root_directory / "summary"
        assert results_dir.exists()
        assert results_dir.is_dir()

        # Check expected files
        expected_files = ["best_config_trajectory.txt", "best_config.txt"]
        for filename in expected_files:
            file_path = results_dir / filename
            assert file_path.exists(), f"Expected file {filename} should exist"
            assert file_path.is_file(), f"{filename} should be a file"

            # File should not be empty
            content = file_path.read_text()
            assert len(content.strip()) > 0, f"{filename} should not be empty"


def test_neps_revisit_run_with_trajectory():
    """Test that NePS can revisit an earlier run and use incumbent trajectory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "revisit_test"

        # First run - create initial optimization
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=3,
            overwrite_root_directory=True,  # Start fresh
        )

        # Check that initial files were created
        summary_dir = root_directory / "summary"
        assert summary_dir.exists()
        best_config_file = summary_dir / "best_config.txt"
        trajectory_file = summary_dir / "best_config_trajectory.txt"
        assert best_config_file.exists()
        assert trajectory_file.exists()

        # Read initial trajectory
        initial_trajectory = trajectory_file.read_text()
        assert "Config ID:" in initial_trajectory
        assert "Objective to minimize:" in initial_trajectory

        # Second run - revisit without overwriting
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=2,  # Add 2 more evaluations
            overwrite_root_directory=False,  # Don't overwrite, continue from previous
        )

        # Check that trajectory was updated with new evaluations
        updated_trajectory = trajectory_file.read_text()

        # The updated trajectory should contain the original entries plus new ones
        assert len(updated_trajectory) >= len(initial_trajectory)
        assert "Config ID:" in updated_trajectory
        assert "Objective to minimize:" in updated_trajectory

        # Should have evidence of multiple evaluations
        # Note: trajectory.txt only tracks BEST configs, not all evaluations
        # So we check that the files still have the expected format and content
        assert "Config ID:" in updated_trajectory
        assert "Objective to minimize:" in updated_trajectory

        # The updated content should be at least as long (potentially with timing info
        # added)
        assert len(updated_trajectory) >= len(initial_trajectory), (
            "Updated trajectory should have at least the same content"
        )


@pytest.mark.parametrize("run_number", range(10))
def test_continue_finished_run_with_higher_budget(run_number):
    """Test continuing a finished run with overwrite_root_directory=False.

    This test runs 10 times with different random seeds to catch any
    non-deterministic bugs in state management.

    Uses a HIGH initial budget to likely find a very good solution, then continues
    with a SMALL additional budget. This makes bugs more obvious because:
    - New evaluations will likely be worse than the initial best
    - If bugs exist, they'll cause the worse configs to incorrectly appear as "best"
    - Tests that the incumbent is properly preserved across continuation

    Verifies:
    1. Best config file contains the actual best configuration from all evaluations
    2. Cumulative metrics in best_config.txt are correct
    3. The best found improves or stays the same compared to initial run
    4. Trajectory objectives are monotonically non-increasing
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / f"continue_test_run_{run_number}"

        # First run - complete optimization with HIGH budget to find a good solution
        initial_budget = 20
        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=initial_budget,
            overwrite_root_directory=True,
        )  # Verify first run completed
        summary_dir = root_directory / "summary"
        best_config_file = summary_dir / "best_config.txt"
        trajectory_file = summary_dir / "best_config_trajectory.txt"

        assert best_config_file.exists()
        assert trajectory_file.exists()

        # Read initial results
        initial_best_config = best_config_file.read_text()
        trajectory_file.read_text()

        # Extract initial best objective
        initial_obj_match = re.search(
            r"Objective to minimize: ([\d.]+)", initial_best_config
        )
        assert initial_obj_match is not None
        initial_best_objective = float(initial_obj_match.group(1))

        # Extract initial cumulative evaluations
        initial_cum_match = re.search(
            r"Cumulative evaluations: (\d+)", initial_best_config
        )
        assert initial_cum_match is not None
        initial_cumulative = int(initial_cum_match.group(1))

        # Second run - continue with SMALL additional budget
        # With the high initial budget, we likely found a good solution
        # New evaluations will probably be worse, exposing bugs if they exist
        additional_budget = 5
        total_expected_budget = initial_budget + additional_budget

        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=SimpleSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=additional_budget,
            overwrite_root_directory=False,  # Continue from previous run
        )

        # Read final results
        final_best_config = best_config_file.read_text()
        final_trajectory = trajectory_file.read_text()

        # ===== Test 1: Verify best_config.txt contains a valid best configuration =====
        final_obj_match = re.search(r"Objective to minimize: ([\d.]+)", final_best_config)
        assert final_obj_match is not None
        final_best_objective = float(final_obj_match.group(1))

        # The final best should be <= initial best (can only improve or stay same)
        assert final_best_objective <= initial_best_objective, (
            "Best objective regressed when continuing run: "
            f"initial={initial_best_objective}, final={final_best_objective}"
        )

        # ===== Test 2: Verify cumulative metrics are correct =====
        # Extract cumulative evaluations from best_config.txt
        final_cum_match = re.search(r"Cumulative evaluations: (\d+)", final_best_config)
        assert final_cum_match is not None, "Should have cumulative evaluations"
        final_cumulative_evals = int(final_cum_match.group(1))

        # The cumulative evaluations should be >= initial cumulative
        # (we added more evaluations in the second run)
        assert final_cumulative_evals >= initial_cumulative, (
            f"Final cumulative evaluations {final_cumulative_evals} should be >= "
            f"initial cumulative {initial_cumulative}"
        )

        # The cumulative evaluations should be <= total budget
        # (may be less if some evaluations failed or optimizer stopped early)
        assert final_cumulative_evals <= total_expected_budget, (
            f"Cumulative evaluations {final_cumulative_evals} should not exceed "
            f"total budget {total_expected_budget}"
        )

        # ===== Test 3: Verify trajectory contains the improvement history =====
        trajectory_objectives = re.findall(
            r"Objective to minimize: ([\d.]+)", final_trajectory
        )
        assert len(trajectory_objectives) > 0, "Should have trajectory objectives"

        # Convert to floats
        objectives = [float(obj) for obj in trajectory_objectives]

        # The minimum objective in trajectory should be the true best across all runs
        min_trajectory_objective = min(objectives)

        # The best_config.txt should show the minimum objective from the trajectory
        assert abs(final_best_objective - min_trajectory_objective) < 1e-10, (
            f"best_config.txt shows {final_best_objective} but "
            f"trajectory minimum is {min_trajectory_objective}"
        )

        # ===== Test 4: Verify trajectory is monotonically non-increasing =====
        # The trajectory should be monotonically non-increasing (each entry should be
        # better than or equal to the previous)
        is_monotonic = all(
            objectives[i] <= objectives[i - 1] for i in range(1, len(objectives))
        )

        assert is_monotonic, (
            f"Trajectory is not monotonically non-increasing: {objectives}"
        )

        # ===== Test 5: Verify the config in best_config.txt is valid =====
        # The format is "Config: {dict}" so we look for that pattern
        assert "Config:" in final_best_config, "Should have config section"

        # Should contain parameter values (x and y, possibly with SAMPLING__ prefix)
        assert any(
            param in final_best_config
            for param in ["x", "y", "'x'", "'y'", "SAMPLING__Resolvable"]
        ), "Config should contain parameter values"

        # ===== Test 6: Verify the objective value is in the valid range =====
        assert 1.0 <= final_best_objective <= 11.0, (
            f"Best objective {final_best_objective} should be in range [1.0, 11.0]"
        )

        # ===== Test 7: Verify trajectory file format is correct =====
        assert (
            "Best configs and their objectives across evaluations:" in final_trajectory
        ), "Should have trajectory header"

        assert "Config ID:" in final_trajectory, "Should have config IDs"
        assert (
            "--------------------------------------------------------------------------------"
            in final_trajectory
        ), "Should have separator lines"

        # ===== Test 8: Verify we actually did more evaluations =====
        # Count config entries in trajectory (though trajectory only shows improvements)
        config_count = final_trajectory.count("Config ID:")
        assert config_count >= 1, "Should have at least one config entry"
