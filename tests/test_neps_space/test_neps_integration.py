from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import pytest

import neps
import neps.optimizers
from neps.optimizers import algorithms
from neps.space.neps_spaces.neps_space import (
    check_neps_space_compatibility,
    convert_classic_to_neps_search_space,
    convert_neps_to_classic_search_space,
)
from neps.space.neps_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    Operation,
    PipelineSpace,
    Resampled,
)


def hyperparameter_pipeline_to_optimize(
    float1: float,
    float2: float,
    categorical: int,
    integer1: int,
    integer2: int,
):
    assert isinstance(float1, float)
    assert isinstance(float2, float)
    assert isinstance(categorical, int)
    assert isinstance(integer1, int)
    assert isinstance(integer2, int)

    objective_to_minimize = -float(float1 + float2 + categorical + integer1 + integer2)
    assert isinstance(objective_to_minimize, float)

    return objective_to_minimize


class DemoHyperparameterSpace(PipelineSpace):
    float1 = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )


class DemoHyperparameterWithFidelitySpace(PipelineSpace):
    float1 = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Fidelity(
        Integer(
            min_value=1,
            max_value=1000,
        ),
    )


class DemoHyperparameterComplexSpace(PipelineSpace):
    _small_float = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    _big_float = Float(
        min_value=10,
        max_value=100,
        prior=20,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )

    float1 = Categorical(
        choices=(
            Resampled(_small_float),
            Resampled(_big_float),
        ),
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Categorical(
        choices=(
            Resampled(_small_float),
            Resampled(_big_float),
            float1,
        ),
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        partial(algorithms.neps_random_search, ignore_fidelity=True),
        partial(algorithms.complex_random_search, ignore_fidelity=True),
    ],
)
def test_hyperparameter_demo(optimizer):
    pipeline_space = DemoHyperparameterSpace()
    root_directory = f"/tests_tmpdir/test_neps_spaces/results/hyperparameter_demo__{optimizer.func.__name__}"

    neps.run(
        evaluate_pipeline=hyperparameter_pipeline_to_optimize,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        evaluations_to_spend=10,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    "optimizer",
    [
        partial(algorithms.neps_random_search, ignore_fidelity=True),
        partial(algorithms.complex_random_search, ignore_fidelity=True),
    ],
)
def test_hyperparameter_with_fidelity_demo(optimizer):
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"/tests_tmpdir/test_neps_spaces/results/hyperparameter_with_fidelity_demo__{optimizer.func.__name__}"

    neps.run(
        evaluate_pipeline=hyperparameter_pipeline_to_optimize,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        evaluations_to_spend=10,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    "optimizer",
    [
        partial(algorithms.neps_random_search, ignore_fidelity=True),
        partial(algorithms.complex_random_search, ignore_fidelity=True),
    ],
)
def test_hyperparameter_complex_demo(optimizer):
    pipeline_space = DemoHyperparameterComplexSpace()
    root_directory = f"/tests_tmpdir/test_neps_spaces/results/hyperparameter_complex_demo__{optimizer.func.__name__}"

    neps.run(
        evaluate_pipeline=hyperparameter_pipeline_to_optimize,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        overwrite_root_directory=True,
        evaluations_to_spend=10,
    )
    neps.status(root_directory, print_summary=True)


# -----------------------------------------


class Model:
    """A simple model that takes an inner function and a factor,
    multiplies the result of the inner function by the factor.
    """

    def __init__(
        self,
        inner_function: Callable[[Sequence[float]], float],
        factor: float,
    ):
        """Initialize the model with an inner function and a factor."""
        self.inner_function = inner_function
        self.factor = factor

    def __call__(self, values: Sequence[float]) -> float:
        return self.factor * self.inner_function(values)


class Sum:
    """A simple inner function that sums the values."""

    def __call__(self, values: Sequence[float]) -> float:
        return sum(values)


class MultipliedSum:
    """An inner function that sums the values and multiplies the result by a factor."""

    def __init__(self, factor: float):
        """Initialize the multiplied sum with a factor."""
        self.factor = factor

    def __call__(self, values: Sequence[float]) -> float:
        return self.factor * sum(values)


def operation_pipeline_to_optimize(model: Model, some_hp: str):
    assert isinstance(model, Model)
    assert isinstance(model.factor, float)
    assert isinstance(model.inner_function, Sum | MultipliedSum)
    if isinstance(model.inner_function, MultipliedSum):
        assert isinstance(model.inner_function.factor, float)
    assert some_hp in {"hp1", "hp2"}

    values = list(range(1, 21))
    objective_to_minimize = model(values)
    assert isinstance(objective_to_minimize, float)

    return objective_to_minimize


class DemoOperationSpace(PipelineSpace):
    """A demonstration of how to use operations in a search space.
    This space defines a model that can be optimized using different inner functions
    and a factor. The model can be used to evaluate a set of values and return an objective to minimize.
    """

    # The way to sample `factor` values
    _factor = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )

    # Sum
    # Will be equivalent to something like
    #   `Sum()`
    # Could have also been defined using the python `sum` function as
    #   `_sum = space.Operation(operator=lambda: sum)`
    _sum = Operation(operator=Sum)

    # MultipliedSum
    # Will be equivalent to something like
    #   `MultipliedSum(factor=0.2)`
    _multiplied_sum = Operation(
        operator=MultipliedSum,
        kwargs={"factor": Resampled(_factor)},
    )

    # Model
    # Will be equivalent to something like one of
    #   `Model(Sum(), factor=0.1)`
    #   `Model(MultipliedSum(factor=0.2), factor=0.1)`
    _inner_function = Categorical(
        choices=(_sum, _multiplied_sum),
    )
    model = Operation(
        operator=Model,
        args=(_inner_function,),
        kwargs={"factor": Resampled(_factor)},
    )

    # An additional hyperparameter
    some_hp = Categorical(
        choices=("hp1", "hp2"),
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        algorithms.neps_random_search,
        algorithms.complex_random_search,
    ],
)
def test_operation_demo(optimizer):
    pipeline_space = DemoOperationSpace()
    root_directory = (
        f"/tests_tmpdir/test_neps_spaces/results/operation_demo__{optimizer.__name__}"
    )

    neps.run(
        evaluate_pipeline=operation_pipeline_to_optimize,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        evaluations_to_spend=10,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)


# ===== Extended tests for newer NePS features =====


# Test neps_hyperband with various PipelineSpaces
@pytest.mark.parametrize(
    "optimizer",
    [
        algorithms.neps_hyperband,
    ],
)
def test_neps_hyperband_with_fidelity_demo(optimizer):
    """Test neps_hyperband with a fidelity space."""
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"/tests_tmpdir/test_neps_spaces/results/neps_hyperband_fidelity_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=hyperparameter_pipeline_to_optimize,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        fidelities_to_spend=15,  # Use fidelities_to_spend for multi-fidelity optimizers
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)


# Test PipelineSpace dynamic methods (add, remove, add_prior)
def test_pipeline_space_dynamic_methods():
    """Test PipelineSpace add, remove, and add_prior methods."""

    # Create a basic space
    class BasicSpace(PipelineSpace):
        x = Float(min_value=0.0, max_value=1.0)
        y = Integer(min_value=1, max_value=10)

    space = BasicSpace()

    # Test adding a new parameter
    new_param = Categorical(choices=(True, False))
    space = space.add(new_param, "flag")

    # Verify the parameter was added
    attrs = space.get_attrs()
    assert "flag" in attrs
    assert attrs["flag"] is new_param

    # Test adding a prior to an existing parameter
    space = space.add_prior("x", prior=0.5, prior_confidence=ConfidenceLevel.HIGH)

    # Verify the prior was added
    updated_attrs = space.get_attrs()
    x_param = updated_attrs["x"]
    assert x_param.has_prior
    assert x_param.prior == 0.5
    assert x_param.prior_confidence == ConfidenceLevel.HIGH

    # Test removing a parameter
    space = space.remove("y")

    # Verify the parameter was removed
    final_attrs = space.get_attrs()
    assert "y" not in final_attrs
    assert "x" in final_attrs
    assert "flag" in final_attrs


# Test space conversion functions
def test_space_conversion_functions():
    """Test conversion between classic and NePS spaces."""
    # Create a classic SearchSpace
    classic_space = neps.SearchSpace(
        {
            "x": neps.HPOFloat(0.0, 1.0, prior=0.5, prior_confidence="medium"),
            "y": neps.HPOInteger(1, 10, prior=5, prior_confidence="high"),
            "z": neps.HPOCategorical(["a", "b", "c"], prior="b", prior_confidence="low"),
        }
    )

    # Convert to NePS space
    neps_space = convert_classic_to_neps_search_space(classic_space)
    assert isinstance(neps_space, PipelineSpace)

    # Verify attributes are preserved
    neps_attrs = neps_space.get_attrs()
    assert len(neps_attrs) == 3
    assert all(name in neps_attrs for name in ["x", "y", "z"])

    # Verify types and priors
    assert isinstance(neps_attrs["x"], Float)
    assert neps_attrs["x"].has_prior
    assert neps_attrs["x"].prior == 0.5

    assert isinstance(neps_attrs["y"], Integer)
    assert neps_attrs["y"].has_prior
    assert neps_attrs["y"].prior == 5

    assert isinstance(neps_attrs["z"], Categorical)
    assert neps_attrs["z"].has_prior
    assert neps_attrs["z"].prior == 1  # Index of "b" in choices

    # Convert back to classic space
    converted_back = convert_neps_to_classic_search_space(neps_space)
    assert converted_back is not None
    assert isinstance(converted_back, neps.SearchSpace)

    # Verify round-trip conversion preserves structure
    classic_attrs = converted_back.elements
    assert len(classic_attrs) == 3
    assert all(name in classic_attrs for name in ["x", "y", "z"])


# Test algorithm compatibility checking
def test_algorithm_compatibility():
    """Test algorithm compatibility with different space types."""
    # Test NePS-only algorithms
    neps_only_algorithms = [
        algorithms.neps_random_search,
        algorithms.neps_hyperband,
        algorithms.complex_random_search,
    ]

    for algo in neps_only_algorithms:
        compatibility = check_neps_space_compatibility(algo)
        assert compatibility in [
            "neps",
            "both",
        ], f"Algorithm {algo.__name__} should be neps or both compatible"

    # Test classic algorithms that should work with both
    both_compatible_algorithms = [
        algorithms.random_search,
        algorithms.hyperband,
    ]

    for algo in both_compatible_algorithms:
        compatibility = check_neps_space_compatibility(algo)
        assert compatibility in [
            "classic",
            "both",
        ], f"Algorithm {algo.__name__} should be classic or both compatible"


# Test with complex PipelineSpace containing Operations and Resampled
def test_complex_neps_space_features():
    """Test complex NePS space features that cannot be converted to classic."""

    class ComplexNepsSpace(PipelineSpace):
        # Basic parameters
        factor = Float(
            min_value=0.1,
            max_value=2.0,
            prior=1.0,
            prior_confidence=ConfidenceLevel.MEDIUM,
        )

        # Operation with resampled parameters
        operation = Operation(
            operator=lambda x, y: x * y,
            args=(factor, Resampled(factor)),
        )

        # Categorical with operations as choices
        choice = Categorical(
            choices=(operation, factor),
            prior=0,
            prior_confidence=ConfidenceLevel.LOW,
        )

    space = ComplexNepsSpace()

    # This space should NOT be convertible to classic
    converted = convert_neps_to_classic_search_space(space)
    assert converted is None, "Complex NePS space should not be convertible to classic"

    # But should work with NePS-compatible algorithms
    compatibility = check_neps_space_compatibility(algorithms.neps_random_search)
    assert compatibility in ["neps", "both"]


# Test trajectory and metrics functionality
def test_trajectory_and_metrics(tmp_path):
    """Test extended trajectory and best_config functionality."""

    def evaluate_with_metrics(x: float, y: int) -> dict:
        """Evaluation function that returns multiple metrics."""
        return {
            "objective_to_minimize": x + y,
            "accuracy": 1.0 - (x + y) / 11.0,  # Dummy accuracy metric
            "training_time": x * 10,  # Dummy training time
            "memory_usage": y * 100,  # Dummy memory usage
        }

    class MetricsSpace(PipelineSpace):
        x = Float(min_value=0.0, max_value=1.0)
        y = Integer(min_value=1, max_value=10)

    space = MetricsSpace()
    root_directory = tmp_path / "metrics_test"

    # Run optimization
    neps.run(
        evaluate_pipeline=evaluate_with_metrics,
        pipeline_space=space,
        optimizer=algorithms.neps_random_search,
        root_directory=str(root_directory),
        evaluations_to_spend=5,
        overwrite_root_directory=True,
    )

    # Check that trajectory and best_config files exist and contain extended metrics
    trajectory_file = root_directory / "summary" / "best_config_trajectory.txt"
    best_config_file = root_directory / "summary" / "best_config.txt"

    assert trajectory_file.exists(), "Trajectory file should exist"
    assert best_config_file.exists(), "Best config file should exist"

    # Read and verify trajectory contains the standard format (not extended metrics in txt files)
    trajectory_content = trajectory_file.read_text()
    assert "Config ID:" in trajectory_content, "Trajectory should contain Config ID"
    assert "Objective to minimize:" in trajectory_content, (
        "Trajectory should contain objective"
    )
    assert "Cumulative evaluations:" in trajectory_content, (
        "Trajectory should contain cumulative evaluations"
    )

    # Read and verify best config contains the standard format
    best_config_content = best_config_file.read_text()
    assert "Config ID:" in best_config_content, "Best config should contain Config ID"
    assert "Objective to minimize:" in best_config_content, (
        "Best config should contain objective"
    )
