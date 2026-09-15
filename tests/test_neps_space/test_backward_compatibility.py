"""Test that legacy SearchSpace / dict pipeline_space inputs to neps.run are rejected."""

from __future__ import annotations

import tempfile
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


def test_searchspace_is_rejected():
    """Passing a classic SearchSpace to neps.run should raise a ValueError."""
    pipeline_space = SearchSpace(
        {
            "learning_rate": HPOFloat(1e-4, 1e-1, log=True),
            "num_layers": HPOInteger(1, 10),
            "optimizer": HPOCategorical(["adam", "sgd", "rmsprop"]),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "searchspace_rejected_test"

        with pytest.raises(ValueError, match="PipelineSpace"):
            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )


def test_dict_is_rejected():
    """Passing a plain dict to neps.run should raise a ValueError."""
    pipeline_space = {
        "learning_rate": Float(1e-4, 1e-1, log=True),
        "num_layers": Integer(1, 10),
        "optimizer": Categorical(["adam", "sgd", "rmsprop"]),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "dict_rejected_test"

        with pytest.raises(ValueError, match="PipelineSpace"):
            neps.run(
                evaluate_pipeline=simple_evaluation,
                pipeline_space=pipeline_space,
                optimizer=algorithms.neps_random_search,
                root_directory=str(root_directory),
                evaluations_to_spend=3,
                overwrite_root_directory=True,
            )


def test_proper_pipelinespace_still_works():
    """Test that using proper PipelineSpace class works as expected."""

    class TestSpace(PipelineSpace):
        learning_rate = Float(1e-4, 1e-1, log=True)
        num_layers = Integer(1, 10)
        optimizer = Categorical(["adam", "sgd", "rmsprop"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_directory = Path(tmp_dir) / "pipelinespace_test"

        neps.run(
            evaluate_pipeline=simple_evaluation,
            pipeline_space=TestSpace(),
            optimizer=algorithms.neps_random_search,
            root_directory=str(root_directory),
            evaluations_to_spend=3,
            overwrite_root_directory=True,
        )

        assert root_directory.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
