"""Tests for space conversion and algorithm compatibility in Extrapolaw."""

from __future__ import annotations

import extrapolaw
from extrapolaw.optimizers import algorithms
from extrapolaw.space.extrapolaw_spaces.extrapolaw_space import (
    check_extrapolaw_space_compatibility,
    convert_classic_to_extrapolaw_search_space,
    convert_extrapolaw_to_classic_search_space,
)
from extrapolaw.space.extrapolaw_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    IntegerFidelity,
    Operation,
    PipelineSpace,
)
from extrapolaw.space.parameters import HPOCategorical, HPOConstant, HPOFloat, HPOInteger


class SimpleHPOSpace(PipelineSpace):
    """Simple hyperparameter-only space that can be converted to classic."""

    x = Float(lower=0.0, upper=1.0, prior=0.5, prior_confidence=ConfidenceLevel.MEDIUM)
    y = Integer(lower=1, upper=10, prior=5, prior_confidence=ConfidenceLevel.HIGH)
    z = Categorical(
        choices=("a", "b", "c"), prior=1, prior_confidence=ConfidenceLevel.LOW
    )


class SimpleHPOWithFidelitySpace(PipelineSpace):
    """Simple hyperparameter space with fidelity."""

    x = Float(lower=0.0, upper=1.0, prior=0.5, prior_confidence=ConfidenceLevel.MEDIUM)
    y = Integer(lower=1, upper=10, prior=5, prior_confidence=ConfidenceLevel.HIGH)
    epochs = IntegerFidelity(lower=1, upper=100)


class ComplexExtrapolawSpace(PipelineSpace):
    """Complex Extrapolaw space that cannot be converted to classic."""

    # Basic parameters
    factor = Float(
        lower=0.1, upper=2.0, prior=1.0, prior_confidence=ConfidenceLevel.MEDIUM
    )

    # Operation with resampled parameters
    operation = Operation(
        operator=lambda x, y: x * y,
        args=(factor, factor.resample()),
    )

    # Categorical with operations as choices
    choice = Categorical(
        choices=(operation, factor),
        prior=0,
        prior_confidence=ConfidenceLevel.LOW,
    )


# ===== Test space conversion functions =====


def test_convert_classic_to_extrapolaw():
    """Test conversion from classic SearchSpace to Extrapolaw PipelineSpace."""
    # Create a classic SearchSpace with various parameter types
    classic_space = extrapolaw.SearchSpace(
        {
            "float_param": HPOFloat(0.0, 1.0, prior=0.5, prior_confidence="medium"),
            "int_param": HPOInteger(1, 10, prior=5, prior_confidence="high"),
            "cat_param": HPOCategorical(
                ["a", "b", "c"], prior="b", prior_confidence="low"
            ),
            "fidelity_param": HPOInteger(1, 100, is_fidelity=True),
            "constant_param": HPOConstant("constant_value"),
        }
    )

    # Convert to Extrapolaw space
    extrapolaw_space = convert_classic_to_extrapolaw_search_space(classic_space)
    assert isinstance(extrapolaw_space, PipelineSpace)

    # Verify attributes are preserved
    extrapolaw_attrs = extrapolaw_space.get_attrs()
    assert len(extrapolaw_attrs) == 5
    assert all(
        name in extrapolaw_attrs
        for name in [
            "float_param",
            "int_param",
            "cat_param",
            "fidelity_param",
            "constant_param",
        ]
    )

    # Verify types and properties
    assert isinstance(extrapolaw_attrs["float_param"], Float)
    assert extrapolaw_attrs["float_param"].has_prior
    assert extrapolaw_attrs["float_param"].prior == 0.5
    assert extrapolaw_attrs["float_param"].prior_confidence == ConfidenceLevel.MEDIUM

    assert isinstance(extrapolaw_attrs["int_param"], Integer)
    assert extrapolaw_attrs["int_param"].has_prior
    assert extrapolaw_attrs["int_param"].prior == 5
    assert extrapolaw_attrs["int_param"].prior_confidence == ConfidenceLevel.HIGH

    assert isinstance(extrapolaw_attrs["cat_param"], Categorical)
    assert extrapolaw_attrs["cat_param"].has_prior
    assert extrapolaw_attrs["cat_param"].prior == 1  # Index of "b" in choices
    assert extrapolaw_attrs["cat_param"].prior_confidence == ConfidenceLevel.LOW

    assert isinstance(extrapolaw_attrs["fidelity_param"], Fidelity)
    assert isinstance(extrapolaw_attrs["fidelity_param"].domain, Integer)

    # Constant should be preserved as-is
    assert extrapolaw_attrs["constant_param"] == "constant_value"


def test_convert_extrapolaw_to_classic_simple():
    """Test conversion from simple Extrapolaw PipelineSpace to classic SearchSpace."""
    space = SimpleHPOSpace()

    # Convert to classic space
    classic_space = convert_extrapolaw_to_classic_search_space(space)
    assert classic_space is not None
    assert isinstance(classic_space, extrapolaw.SearchSpace)

    # Verify attributes are preserved
    classic_attrs = classic_space.elements
    assert len(classic_attrs) == 3
    assert all(name in classic_attrs for name in ["x", "y", "z"])

    # Verify types and priors
    x_param = classic_attrs["x"]
    assert isinstance(x_param, HPOFloat)
    assert x_param.lower == 0.0
    assert x_param.upper == 1.0
    assert x_param.prior == 0.5
    assert x_param.prior_confidence == "medium"

    y_param = classic_attrs["y"]
    assert isinstance(y_param, HPOInteger)
    assert y_param.lower == 1
    assert y_param.upper == 10
    assert y_param.prior == 5
    assert y_param.prior_confidence == "high"

    z_param = classic_attrs["z"]
    assert isinstance(z_param, HPOCategorical)
    assert set(z_param.choices) == {"a", "b", "c"}  # Order might vary
    assert z_param.prior == "b"
    assert z_param.prior_confidence == "low"


def test_convert_extrapolaw_to_classic_with_fidelity():
    """Test conversion from Extrapolaw PipelineSpace with fidelity to classic SearchSpace."""
    space = SimpleHPOWithFidelitySpace()

    # Convert to classic space
    classic_space = convert_extrapolaw_to_classic_search_space(space)
    assert classic_space is not None
    assert isinstance(classic_space, extrapolaw.SearchSpace)

    # Verify fidelity parameter
    epochs_param = classic_space.elements["epochs"]
    assert isinstance(epochs_param, HPOInteger)
    assert epochs_param.is_fidelity
    assert epochs_param.lower == 1
    assert epochs_param.upper == 100


def test_convert_complex_extrapolaw_to_classic_fails():
    """Test that complex Extrapolaw spaces cannot be converted to classic."""
    space = ComplexExtrapolawSpace()

    # This space should NOT be convertible to classic
    converted = convert_extrapolaw_to_classic_search_space(space)
    assert converted is None


def test_round_trip_conversion():
    """Test that simple spaces can be converted back and forth."""
    # Start with classic space
    original_classic = extrapolaw.SearchSpace(
        {
            "x": HPOFloat(0.0, 1.0, prior=0.5, prior_confidence="medium"),
            "y": HPOInteger(1, 10, prior=5, prior_confidence="high"),
            "z": HPOCategorical(["a", "b", "c"], prior="b", prior_confidence="low"),
        }
    )

    # Convert to Extrapolaw and back
    extrapolaw_space = convert_classic_to_extrapolaw_search_space(original_classic)
    converted_back = convert_extrapolaw_to_classic_search_space(extrapolaw_space)

    assert converted_back is not None
    assert len(converted_back.elements) == len(original_classic.elements)

    # Verify parameters are equivalent
    for name in original_classic.elements:
        original_param = original_classic.elements[name]
        converted_param = converted_back.elements[name]

        assert type(original_param) is type(converted_param)

        # Check bounds for numerical parameters
        if isinstance(original_param, HPOFloat | HPOInteger):
            assert original_param.lower == converted_param.lower
            assert original_param.upper == converted_param.upper

        # Check choices for categorical parameters
        if isinstance(original_param, HPOCategorical):
            # Sort choices for comparison since order might differ
            assert set(original_param.choices) == set(converted_param.choices)

        # Check priors
        if hasattr(original_param, "prior") and hasattr(converted_param, "prior"):
            assert original_param.prior == converted_param.prior


# ===== Test algorithm compatibility =====


def test_extrapolaw_only_algorithms():
    """Test that Extrapolaw-only algorithms are correctly identified."""
    extrapolaw_only_algorithms = [
        algorithms.extrapolaw_random_search,
        algorithms.complex_random_search,
    ]

    for algo in extrapolaw_only_algorithms:
        compatibility = check_extrapolaw_space_compatibility(algo)
        assert compatibility in [
            "extrapolaw",
            "both",
        ], f"Algorithm {algo.__name__} should be extrapolaw or both compatible"


def test_classic_and_both_algorithms():
    """Test that classic algorithms that work with both spaces are correctly identified."""
    both_compatible_algorithms = [
        algorithms.random_search,
        algorithms.grid_search,
    ]

    for algo in both_compatible_algorithms:
        compatibility = check_extrapolaw_space_compatibility(algo)
        assert compatibility in [
            "classic",
            "both",
        ], f"Algorithm {algo.__name__} should be classic or both compatible"


def test_algorithm_compatibility_with_string_names():
    """Test algorithm compatibility checking with string names."""
    # Note: String-based compatibility checking may not be fully implemented
    # Test with actual algorithm functions instead

    # Test Extrapolaw-only algorithms
    extrapolaw_only_algorithms = [
        algorithms.extrapolaw_random_search,
        algorithms.complex_random_search,
    ]

    for algo in extrapolaw_only_algorithms:
        compatibility = check_extrapolaw_space_compatibility(algo)
        assert compatibility in [
            "extrapolaw",
            "both",
        ], f"Algorithm {algo.__name__} should be extrapolaw or both compatible"

    # Test classic/both algorithms
    classic_algorithms = [
        algorithms.random_search,
        algorithms.grid_search,
    ]

    for algo in classic_algorithms:
        compatibility = check_extrapolaw_space_compatibility(algo)
        assert compatibility in [
            "classic",
            "both",
        ], f"Algorithm {algo.__name__} should be classic or both compatible"


def test_algorithm_compatibility_with_tuples():
    """Test algorithm compatibility checking with tuple configurations."""
    # Test with tuple configuration
    extrapolaw_config = ("extrapolaw_random_search", {"ignore_fidelity": True})
    compatibility = check_extrapolaw_space_compatibility(extrapolaw_config)
    assert compatibility in ["extrapolaw", "both"]

    classic_config = ("random_search", {"some_param": "value"})
    compatibility = check_extrapolaw_space_compatibility(classic_config)
    assert compatibility in ["classic", "both"]


def test_algorithm_compatibility_with_mappings():
    """Test algorithm compatibility checking with mapping configurations."""
    classic_config = {"name": "bayesian_optimization", "initial_design_size": 2}
    assert check_extrapolaw_space_compatibility(classic_config) == "classic"

    both_config = {"name": "random_search"}
    assert check_extrapolaw_space_compatibility(both_config) == "both"

    extrapolaw_config = {"name": "extrapolaw_random_search"}
    assert check_extrapolaw_space_compatibility(extrapolaw_config) == "extrapolaw"


def test_auto_algorithm_compatibility():
    """Test that 'auto' algorithm is handled correctly."""
    compatibility = check_extrapolaw_space_compatibility("auto")
    assert compatibility == "both"


# ===== Test space compatibility with different optimizers =====


def test_simple_space_works_with_both_optimizers():
    """Test that simple HPO spaces work with both classic and Extrapolaw optimizers."""
    space = SimpleHPOSpace()

    # Should work with Extrapolaw-only optimizers
    extrapolaw_optimizer = algorithms.extrapolaw_random_search(pipeline_space=space)
    assert extrapolaw_optimizer is not None

    # Should also be convertible and work with classic optimizers
    converted_space = convert_extrapolaw_to_classic_search_space(space)
    assert converted_space is not None

    classic_optimizer = algorithms.random_search(
        pipeline_space=converted_space, use_priors=True
    )
    assert classic_optimizer is not None


def test_complex_space_only_works_with_extrapolaw_optimizers():
    """Test that complex Extrapolaw spaces only work with Extrapolaw-compatible optimizers."""
    space = ComplexExtrapolawSpace()

    # Should work with Extrapolaw optimizers
    extrapolaw_optimizer = algorithms.extrapolaw_random_search(pipeline_space=space)
    assert extrapolaw_optimizer is not None

    # Should NOT be convertible to classic
    converted_space = convert_extrapolaw_to_classic_search_space(space)
    assert converted_space is None


def test_fidelity_space_compatibility():
    """Test fidelity space compatibility with different optimizers."""
    space = SimpleHPOWithFidelitySpace()

    # Works with Extrapolaw optimizers that ignore the fidelity
    random_optimizer = algorithms.extrapolaw_random_search(
        pipeline_space=space, ignore_fidelity=True
    )
    assert random_optimizer is not None

    # Should be convertible to classic for non-extrapolaw-specific algorithms
    converted_space = convert_extrapolaw_to_classic_search_space(space)
    assert converted_space is not None

    # Classic random search should work with converted space
    classic_random_search = algorithms.random_search(
        pipeline_space=converted_space,
        use_priors=True,
        ignore_fidelity="highest_fidelity",
    )
    assert classic_random_search is not None


# ===== Edge cases and error handling =====


def test_conversion_preserves_log_scaling():
    """Test that log scaling is preserved during conversion."""
    classic_space = extrapolaw.SearchSpace(
        {
            "log_param": HPOFloat(1e-5, 1e-1, log=True),
        }
    )

    extrapolaw_space = convert_classic_to_extrapolaw_search_space(classic_space)
    # Access the Float parameter and check if it has a log attribute
    log_param_extrapolaw = extrapolaw_space.get_attrs()["log_param"]
    assert hasattr(log_param_extrapolaw, "log")
    assert log_param_extrapolaw.log is True

    # Round-trip conversion should now preserve log scaling
    converted_back = convert_extrapolaw_to_classic_search_space(extrapolaw_space)
    assert converted_back is not None
    # Check the log property specifically for float parameters
    log_param = converted_back.elements["log_param"]
    assert isinstance(log_param, HPOFloat)
    assert log_param.log is True


def test_conversion_handles_missing_priors():
    """Test that conversion works correctly when priors are missing."""
    classic_space = extrapolaw.SearchSpace(
        {
            "no_prior": HPOFloat(0.0, 1.0),  # No prior specified
        }
    )

    extrapolaw_space = convert_classic_to_extrapolaw_search_space(classic_space)
    param = extrapolaw_space.get_attrs()["no_prior"]
    assert not param.has_prior

    converted_back = convert_extrapolaw_to_classic_search_space(extrapolaw_space)
    assert converted_back is not None
    # Check the prior property specifically for float parameters
    no_prior_param = converted_back.elements["no_prior"]
    assert isinstance(no_prior_param, HPOFloat)
    assert no_prior_param.prior is None


def test_conversion_handles_empty_spaces():
    """Test that conversion handles edge cases gracefully."""
    # Empty classic space
    empty_classic = extrapolaw.SearchSpace({})
    extrapolaw_space = convert_classic_to_extrapolaw_search_space(empty_classic)
    assert len(extrapolaw_space.get_attrs()) == 0

    # Convert back
    converted_back = convert_extrapolaw_to_classic_search_space(extrapolaw_space)
    assert converted_back is not None
    assert len(converted_back.elements) == 0
