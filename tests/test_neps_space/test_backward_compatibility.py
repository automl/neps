"""Test backward compatibility with old SearchSpace and dict-based spaces."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest

import neps
from neps.optimizers import algorithms
from neps.space import HPOCategorical, HPOFloat, HPOInteger, SearchSpace
from neps.space.neps_spaces.parameters import (
    Categorical,
    Float,
    Integer,
    PipelineSpace,
)


def simple_evaluation(learning_rate: float, num_layers: int, optimizer: str) -> float:
    """Simple evaluation function."""
    return learning_rate * num_layers + (0.1 if optimizer == "adam" else 0.2)


def test_searchspace_with_hpo_parameters():
    """Test SearchSpace with old HPO* parameters still works."""
    pipeline_space = SearchSpace(
        {
            "learning_rate": HPOFloat(1e-4, 1e-1, log=True),
            "num_layers": HPOInteger(1, 10),
            "optimizer": HPOCategorical(["adam", "sgd", "rmsprop"]),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "searchspace_hpo_test"

        # Should warn about using SearchSpace instead of PipelineSpace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about SearchSpace
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "SearchSpace" in str(warning.message)
                for warning in w
            ), "Should warn about using SearchSpace"

        assert root_directory.exists()


def test_searchspace_with_new_parameters():
    """Test SearchSpace with new PipelineSpace parameters (Float, Integer, Categorical)."""
    pipeline_space = SearchSpace(
        {
            "learning_rate": Float(1e-4, 1e-1, log=True),
            "num_layers": Integer(1, 10),
            "optimizer": Categorical(["adam", "sgd", "rmsprop"]),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "searchspace_new_test"

        # Should warn about using SearchSpace instead of PipelineSpace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about SearchSpace
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "SearchSpace" in str(warning.message)
                for warning in w
            ), "Should warn about using SearchSpace"

        assert root_directory.exists()


def test_dict_with_hpo_parameters():
    """Test dict-based space with old HPO* parameters still works."""
    pipeline_space = {
        "learning_rate": HPOFloat(1e-4, 1e-1, log=True),
        "num_layers": HPOInteger(1, 10),
        "optimizer": HPOCategorical(["adam", "sgd", "rmsprop"]),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "dict_hpo_test"

        # Should warn about using dict instead of PipelineSpace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about dict
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "dictionary" in str(warning.message).lower()
                for warning in w
            ), "Should warn about using dict"

        assert root_directory.exists()


def test_dict_with_new_parameters():
    """Test dict-based space with new PipelineSpace parameters (Float, Integer, Categorical)."""
    pipeline_space = {
        "learning_rate": Float(1e-4, 1e-1, log=True),
        "num_layers": Integer(1, 10),
        "optimizer": Categorical(["adam", "sgd", "rmsprop"]),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "dict_new_test"

        # Should warn about using dict instead of PipelineSpace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about dict
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "dictionary" in str(warning.message).lower()
                for warning in w
            ), "Should warn about using dict"

        assert root_directory.exists()


def test_searchspace_with_is_fidelity():
    """Test SearchSpace with is_fidelity parameter (old style) still works."""
    pipeline_space = SearchSpace(
        {
            "learning_rate": Float(1e-4, 1e-1, log=True),
            "num_layers": Integer(1, 10),
            "optimizer": Categorical(["adam", "sgd", "rmsprop"]),
            "epochs": Integer(1, 100, is_fidelity=True),
        }
    )

    def fidelity_evaluation(
        learning_rate: float,
        num_layers: int,
        optimizer: str,
        epochs: int,
    ) -> float:
        """Evaluation with fidelity."""
        return learning_rate * num_layers * epochs / 100

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "searchspace_fidelity_test"

        # Should work without errors (just warn about SearchSpace)
        # Use neps_hyperband which supports fidelities
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=fidelity_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_hyperband,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about SearchSpace
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "SearchSpace" in str(warning.message)
                for warning in w
            ), "Should warn about using SearchSpace"

        assert root_directory.exists()


def test_dict_with_is_fidelity():
    """Test dict-based space with is_fidelity parameter (old style) still works."""
    pipeline_space = {
        "learning_rate": Float(1e-4, 1e-1, log=True),
        "num_layers": Integer(1, 10),
        "optimizer": Categorical(["adam", "sgd", "rmsprop"]),
        "epochs": Integer(1, 100, is_fidelity=True),
    }

    def fidelity_evaluation(
        learning_rate: float,
        num_layers: int,
        optimizer: str,
        epochs: int,
    ) -> float:
        """Evaluation with fidelity."""
        return learning_rate * num_layers * epochs / 100

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "dict_fidelity_test"

        # Should work without errors (just warn about dict)
        # Use neps_hyperband which supports fidelities
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=fidelity_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_hyperband,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should get deprecation warning about dict
            assert any(
                issubclass(warning.category, DeprecationWarning)
                and "dictionary" in str(warning.message).lower()
                for warning in w
            ), "Should warn about using dict"

        assert root_directory.exists()


def test_proper_pipelinespace_no_warnings():
    """Test that using proper PipelineSpace class doesn't trigger warnings."""

    class TestSpace(PipelineSpace):
        learning_rate = Float(1e-4, 1e-1, log=True)
        num_layers = Integer(1, 10)
        optimizer = Categorical(["adam", "sgd", "rmsprop"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "pipelinespace_test"

        # Should NOT warn when using proper PipelineSpace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=TestSpace(),
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )

            # Should NOT get deprecation warning about SearchSpace or dict
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and (
                    "SearchSpace" in str(warning.message)
                    or "dictionary" in str(warning.message).lower()
                )
            ]
            assert len(deprecation_warnings) == 0, (
                "Should not warn when using proper PipelineSpace, "
                f"but got: {[str(w.message) for w in deprecation_warnings]}"
            )

        assert root_directory.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
