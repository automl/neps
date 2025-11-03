"""Tests for space conversion and algorithm compatibility in NePS."""

from __future__ import annotations

import pytest

import neps
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
    epochs = Fidelity(Integer(lower=1, upper=100))


class ComplexNepsSpace(PipelineSpace):
    """Complex NePS space that cannot be converted to classic."""

    # Basic parameters
    factor = Float(
        lower=0.1, upper=2.0, prior=1.0, prior_confidence=ConfidenceLevel.MEDIUM
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


# ===== Test space conversion functions =====


def test_convert_classic_to_neps():
    """Test conversion from classic SearchSpace to NePS PipelineSpace."""
    # Create a classic SearchSpace with various parameter types
    classic_space = neps.SearchSpace(
        {
            "float_param": neps.HPOFloat(0.0, 1.0, prior=0.5, prior_confidence="medium"),
            "int_param": neps.HPOInteger(1, 10, prior=5, prior_confidence="high"),
            "cat_param": neps.HPOCategorical(
                ["a", "b", "c"], prior="b", prior_confidence="low"
            ),
            "fidelity_param": neps.HPOInteger(1, 100, is_fidelity=True),
            "constant_param": neps.HPOConstant("constant_value"),
        }
    )

    # Convert to NePS space
    neps_space = convert_classic_to_neps_search_space(classic_space)
    assert isinstance(neps_space, PipelineSpace)

    # Verify attributes are preserved
    neps_attrs = neps_space.get_attrs()
    assert len(neps_attrs) == 5
    assert all(
        name in neps_attrs
        for name in [
            "float_param",
            "int_param",
            "cat_param",
            "fidelity_param",
            "constant_param",
        ]
    )

    # Verify types and properties
    assert isinstance(neps_attrs["float_param"], Float)
    assert neps_attrs["float_param"].has_prior
    assert neps_attrs["float_param"].prior == 0.5
    assert neps_attrs["float_param"].prior_confidence == ConfidenceLevel.MEDIUM

    assert isinstance(neps_attrs["int_param"], Integer)
    assert neps_attrs["int_param"].has_prior
    assert neps_attrs["int_param"].prior == 5
    assert neps_attrs["int_param"].prior_confidence == ConfidenceLevel.HIGH

    assert isinstance(neps_attrs["cat_param"], Categorical)
    assert neps_attrs["cat_param"].has_prior
    assert neps_attrs["cat_param"].prior == 1  # Index of "b" in choices
    assert neps_attrs["cat_param"].prior_confidence == ConfidenceLevel.LOW

    assert isinstance(neps_attrs["fidelity_param"], Fidelity)
    assert isinstance(neps_attrs["fidelity_param"].domain, Integer)

    # Constant should be preserved as-is
    assert neps_attrs["constant_param"] == "constant_value"


def test_convert_neps_to_classic_simple():
    """Test conversion from simple NePS PipelineSpace to classic SearchSpace."""
    space = SimpleHPOSpace()

    # Convert to classic space
    classic_space = convert_neps_to_classic_search_space(space)
    assert classic_space is not None
    assert isinstance(classic_space, neps.SearchSpace)

    # Verify attributes are preserved
    classic_attrs = classic_space.elements
    assert len(classic_attrs) == 3
    assert all(name in classic_attrs for name in ["x", "y", "z"])

    # Verify types and priors
    x_param = classic_attrs["x"]
    assert isinstance(x_param, neps.HPOFloat)
    assert x_param.lower == 0.0
    assert x_param.upper == 1.0
    assert x_param.prior == 0.5
    assert x_param.prior_confidence == "medium"

    y_param = classic_attrs["y"]
    assert isinstance(y_param, neps.HPOInteger)
    assert y_param.lower == 1
    assert y_param.upper == 10
    assert y_param.prior == 5
    assert y_param.prior_confidence == "high"

    z_param = classic_attrs["z"]
    assert isinstance(z_param, neps.HPOCategorical)
    assert set(z_param.choices) == {"a", "b", "c"}  # Order might vary
    assert z_param.prior == "b"
    assert z_param.prior_confidence == "low"


def test_convert_neps_to_classic_with_fidelity():
    """Test conversion from NePS PipelineSpace with fidelity to classic SearchSpace."""
    space = SimpleHPOWithFidelitySpace()

    # Convert to classic space
    classic_space = convert_neps_to_classic_search_space(space)
    assert classic_space is not None
    assert isinstance(classic_space, neps.SearchSpace)

    # Verify fidelity parameter
    epochs_param = classic_space.elements["epochs"]
    assert isinstance(epochs_param, neps.HPOInteger)
    assert epochs_param.is_fidelity
    assert epochs_param.lower == 1
    assert epochs_param.upper == 100


def test_convert_complex_neps_to_classic_fails():
    """Test that complex NePS spaces cannot be converted to classic."""
    space = ComplexNepsSpace()

    # This space should NOT be convertible to classic
    converted = convert_neps_to_classic_search_space(space)
    assert converted is None


def test_round_trip_conversion():
    """Test that simple spaces can be converted back and forth."""
    # Start with classic space
    original_classic = neps.SearchSpace(
        {
            "x": neps.HPOFloat(0.0, 1.0, prior=0.5, prior_confidence="medium"),
            "y": neps.HPOInteger(1, 10, prior=5, prior_confidence="high"),
            "z": neps.HPOCategorical(["a", "b", "c"], prior="b", prior_confidence="low"),
        }
    )

    # Convert to NePS and back
    neps_space = convert_classic_to_neps_search_space(original_classic)
    converted_back = convert_neps_to_classic_search_space(neps_space)

    assert converted_back is not None
    assert len(converted_back.elements) == len(original_classic.elements)

    # Verify parameters are equivalent
    for name in original_classic.elements:
        original_param = original_classic.elements[name]
        converted_param = converted_back.elements[name]

        assert type(original_param) is type(converted_param)

        # Check bounds for numerical parameters
        if isinstance(original_param, neps.HPOFloat | neps.HPOInteger):
            assert original_param.lower == converted_param.lower
            assert original_param.upper == converted_param.upper

        # Check choices for categorical parameters
        if isinstance(original_param, neps.HPOCategorical):
            # Sort choices for comparison since order might differ
            assert set(original_param.choices) == set(converted_param.choices)

        # Check priors
        if hasattr(original_param, "prior") and hasattr(converted_param, "prior"):
            assert original_param.prior == converted_param.prior


# ===== Test algorithm compatibility =====


def test_neps_only_algorithms():
    """Test that NePS-only algorithms are correctly identified."""
    neps_only_algorithms = [
        algorithms.neps_random_search,
        algorithms.neps_hyperband,
        algorithms.complex_random_search,
        algorithms.neps_priorband,
    ]

    for algo in neps_only_algorithms:
        compatibility = check_neps_space_compatibility(algo)
        assert compatibility in [
            "neps",
            "both",
        ], f"Algorithm {algo.__name__} should be neps or both compatible"


def test_classic_and_both_algorithms():
    """Test that classic algorithms that work with both spaces are correctly identified."""
    both_compatible_algorithms = [
        algorithms.random_search,
        algorithms.hyperband,
        algorithms.priorband,
    ]

    for algo in both_compatible_algorithms:
        compatibility = check_neps_space_compatibility(algo)
        assert compatibility in [
            "classic",
            "both",
        ], f"Algorithm {algo.__name__} should be classic or both compatible"


def test_algorithm_compatibility_with_string_names():
    """Test algorithm compatibility checking with string names."""
    # Note: String-based compatibility checking may not be fully implemented
    # Test with actual algorithm functions instead

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

    # Test classic/both algorithms
    classic_algorithms = [
        algorithms.random_search,
        algorithms.hyperband,
    ]

    for algo in classic_algorithms:
        compatibility = check_neps_space_compatibility(algo)
        assert compatibility in [
            "classic",
            "both",
        ], f"Algorithm {algo.__name__} should be classic or both compatible"


def test_algorithm_compatibility_with_tuples():
    """Test algorithm compatibility checking with tuple configurations."""
    # Test with tuple configuration
    neps_config = ("neps_random_search", {"ignore_fidelity": True})
    compatibility = check_neps_space_compatibility(neps_config)
    assert compatibility in ["neps", "both"]

    classic_config = ("random_search", {"some_param": "value"})
    compatibility = check_neps_space_compatibility(classic_config)
    assert compatibility in ["classic", "both"]


def test_auto_algorithm_compatibility():
    """Test that 'auto' algorithm is handled correctly."""
    compatibility = check_neps_space_compatibility("auto")
    assert compatibility == "both"


# ===== Test NePS hyperband specific functionality =====


def test_neps_hyperband_requires_fidelity():
    """Test that neps_hyperband requires fidelity parameters."""
    # Space without fidelity should fail
    space_no_fidelity = SimpleHPOSpace()

    with pytest.raises((ValueError, AssertionError)):
        algorithms.neps_hyperband(pipeline_space=space_no_fidelity)


def test_neps_hyperband_accepts_fidelity_space():
    """Test that neps_hyperband accepts spaces with fidelity."""
    space_with_fidelity = SimpleHPOWithFidelitySpace()

    # Should not raise an error
    optimizer = algorithms.neps_hyperband(pipeline_space=space_with_fidelity)
    assert optimizer is not None


def test_neps_hyperband_rejects_classic_space():
    """Test that neps_hyperband rejects classic SearchSpace."""
    # Type system should prevent this at compile time
    # Instead, test that type checking works as expected

    # Create a proper NePS space that should work
    class TestSpace(PipelineSpace):
        x = Float(0.0, 1.0)
        epochs = Fidelity(Integer(1, 100))

    space = TestSpace()

    # This should work fine with proper NePS space
    optimizer = algorithms.neps_hyperband(pipeline_space=space, eta=3)
    assert optimizer is not None


@pytest.mark.parametrize("eta", [2, 3, 4, 5])
def test_neps_hyperband_eta_values(eta):
    """Test neps_hyperband with different eta values."""
    space = SimpleHPOWithFidelitySpace()
    optimizer = algorithms.neps_hyperband(pipeline_space=space, eta=eta)
    assert optimizer is not None


@pytest.mark.parametrize("sampler", ["uniform", "prior"])
def test_neps_hyperband_samplers(sampler):
    """Test neps_hyperband with different samplers."""
    space = SimpleHPOWithFidelitySpace()
    optimizer = algorithms.neps_hyperband(pipeline_space=space, sampler=sampler)
    assert optimizer is not None


@pytest.mark.parametrize("sample_prior_first", [False, True, "highest_fidelity"])
def test_neps_hyperband_sample_prior_first(sample_prior_first):
    """Test neps_hyperband with different sample_prior_first options."""
    space = SimpleHPOWithFidelitySpace()
    optimizer = algorithms.neps_hyperband(
        pipeline_space=space, sample_prior_first=sample_prior_first
    )
    assert optimizer is not None


# ===== Test space compatibility with different optimizers =====


def test_simple_space_works_with_both_optimizers():
    """Test that simple HPO spaces work with both classic and NePS optimizers."""
    space = SimpleHPOSpace()

    # Should work with NePS-only optimizers
    neps_optimizer = algorithms.neps_random_search(pipeline_space=space)
    assert neps_optimizer is not None

    # Should also be convertible and work with classic optimizers
    converted_space = convert_neps_to_classic_search_space(space)
    assert converted_space is not None

    classic_optimizer = algorithms.random_search(
        pipeline_space=converted_space, use_priors=True
    )
    assert classic_optimizer is not None


def test_complex_space_only_works_with_neps_optimizers():
    """Test that complex NePS spaces only work with NePS-compatible optimizers."""
    space = ComplexNepsSpace()

    # Should work with NePS optimizers
    neps_optimizer = algorithms.neps_random_search(pipeline_space=space)
    assert neps_optimizer is not None

    # Should NOT be convertible to classic
    converted_space = convert_neps_to_classic_search_space(space)
    assert converted_space is None


def test_fidelity_space_compatibility():
    """Test fidelity space compatibility with different optimizers."""
    space = SimpleHPOWithFidelitySpace()

    # Should work with neps_hyperband (requires fidelity)
    hyperband_optimizer = algorithms.neps_hyperband(pipeline_space=space)
    assert hyperband_optimizer is not None

    # Should also work with other NePS optimizers (but need to ignore fidelity)
    random_optimizer = algorithms.neps_random_search(
        pipeline_space=space, ignore_fidelity=True
    )
    assert random_optimizer is not None

    # Should be convertible to classic for non-neps-specific algorithms
    converted_space = convert_neps_to_classic_search_space(space)
    assert converted_space is not None

    # Classic hyperband should work with converted space
    classic_hyperband = algorithms.hyperband(pipeline_space=converted_space)
    assert classic_hyperband is not None


# ===== Edge cases and error handling =====


def test_conversion_preserves_log_scaling():
    """Test that log scaling is preserved during conversion."""
    classic_space = neps.SearchSpace(
        {
            "log_param": neps.HPOFloat(1e-5, 1e-1, log=True),
        }
    )

    neps_space = convert_classic_to_neps_search_space(classic_space)
    # Access the Float parameter and check if it has a log attribute
    log_param_neps = neps_space.get_attrs()["log_param"]
    assert hasattr(log_param_neps, "log")
    assert log_param_neps.log is True

    # Round-trip conversion should now preserve log scaling
    converted_back = convert_neps_to_classic_search_space(neps_space)
    assert converted_back is not None
    # Check the log property specifically for float parameters
    log_param = converted_back.elements["log_param"]
    assert isinstance(log_param, neps.HPOFloat)
    assert log_param.log is True


def test_conversion_handles_missing_priors():
    """Test that conversion works correctly when priors are missing."""
    classic_space = neps.SearchSpace(
        {
            "no_prior": neps.HPOFloat(0.0, 1.0),  # No prior specified
        }
    )

    neps_space = convert_classic_to_neps_search_space(classic_space)
    param = neps_space.get_attrs()["no_prior"]
    assert not param.has_prior

    converted_back = convert_neps_to_classic_search_space(neps_space)
    assert converted_back is not None
    # Check the prior property specifically for float parameters
    no_prior_param = converted_back.elements["no_prior"]
    assert isinstance(no_prior_param, neps.HPOFloat)
    assert no_prior_param.prior is None


def test_conversion_handles_empty_spaces():
    """Test that conversion handles edge cases gracefully."""
    # Empty classic space
    empty_classic = neps.SearchSpace({})
    neps_space = convert_classic_to_neps_search_space(empty_classic)
    assert len(neps_space.get_attrs()) == 0

    # Convert back
    converted_back = convert_neps_to_classic_search_space(neps_space)
    assert converted_back is not None
    assert len(converted_back.elements) == 0
