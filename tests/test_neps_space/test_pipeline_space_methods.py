"""Tests for PipelineSpace dynamic methods (add, remove, add_prior)."""

from __future__ import annotations

import pytest

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


class BasicSpace(PipelineSpace):
    """Basic space for testing dynamic methods."""

    x = Float(lower=0.0, upper=1.0)
    y = Integer(lower=1, upper=10)
    z = Categorical(choices=("a", "b", "c"))


class SpaceWithPriors(PipelineSpace):
    """Space with existing priors for testing."""

    x = Float(lower=0.0, upper=1.0, prior=0.5, prior_confidence=ConfidenceLevel.MEDIUM)
    y = Integer(lower=1, upper=10, prior=5, prior_confidence=ConfidenceLevel.HIGH)
    z = Categorical(
        choices=("a", "b", "c"), prior=1, prior_confidence=ConfidenceLevel.LOW
    )


# ===== Test add method =====


def test_add_method_basic():
    """Test basic functionality of the add method."""
    space = BasicSpace()
    original_attrs = space.get_attrs()

    # Add a new parameter
    new_param = Float(lower=10.0, upper=20.0)
    updated_space = space.add(new_param, "new_float")

    # Original space should be unchanged
    assert space.get_attrs() == original_attrs

    # Updated space should have the new parameter
    updated_attrs = updated_space.get_attrs()
    assert "new_float" in updated_attrs
    assert updated_attrs["new_float"] is new_param
    assert len(updated_attrs) == len(original_attrs) + 1


def test_add_method_different_types():
    """Test adding different parameter types."""
    space = BasicSpace()

    # Add Integer
    space = space.add(Integer(lower=0, upper=100), "new_int")
    assert "new_int" in space.get_attrs()
    assert isinstance(space.get_attrs()["new_int"], Integer)

    # Add Categorical
    space = space.add(Categorical(choices=(True, False)), "new_cat")
    assert "new_cat" in space.get_attrs()
    assert isinstance(space.get_attrs()["new_cat"], Categorical)

    # Add Operation
    op = Operation(operator=lambda x: x * 2, args=(space.get_attrs()["x"],))
    space = space.add(op, "new_op")
    assert "new_op" in space.get_attrs()
    assert isinstance(space.get_attrs()["new_op"], Operation)

    # Add Resampled
    resampled = Resampled(space.get_attrs()["x"])
    space = space.add(resampled, "new_resampled")
    assert "new_resampled" in space.get_attrs()
    assert isinstance(space.get_attrs()["new_resampled"], Resampled)


def test_add_method_with_default_name():
    """Test add method with automatic name generation."""
    space = BasicSpace()
    original_count = len(space.get_attrs())

    # Add without specifying name
    new_param = Float(lower=5.0, upper=15.0)
    updated_space = space.add(new_param)

    updated_attrs = updated_space.get_attrs()
    assert len(updated_attrs) == original_count + 1

    # Should have generated a name like "param_4"
    generated_names = [name for name in updated_attrs if name.startswith("param_")]
    assert len(generated_names) >= 1


def test_add_method_duplicate_parameter():
    """Test adding a parameter with an existing name but same content."""
    space = BasicSpace()

    # Add the same parameter that already exists
    existing_param = space.get_attrs()["x"]
    updated_space = space.add(existing_param, "x")

    # Should work without error
    assert updated_space.get_attrs()["x"] is existing_param


def test_add_method_conflicting_parameter():
    """Test adding a different parameter with an existing name."""
    space = BasicSpace()

    # Try to add a different parameter with existing name
    different_param = Integer(lower=0, upper=5)  # Different from existing "x"

    with pytest.raises(ValueError, match="A different parameter with the name"):
        space.add(different_param, "x")


def test_add_method_chaining():
    """Test chaining multiple add operations."""
    space = BasicSpace()

    # Chain multiple additions
    final_space = (
        space.add(Float(lower=100.0, upper=200.0), "param1")
        .add(Integer(lower=0, upper=50), "param2")
        .add(Categorical(choices=(1, 2, 3)), "param3")
    )

    attrs = final_space.get_attrs()
    assert "param1" in attrs
    assert "param2" in attrs
    assert "param3" in attrs
    assert len(attrs) == 6  # 3 original + 3 new


# ===== Test remove method =====


def test_remove_method_basic():
    """Test basic functionality of the remove method."""
    space = BasicSpace()
    original_attrs = space.get_attrs()

    # Remove a parameter
    updated_space = space.remove("y")

    # Original space should be unchanged
    assert space.get_attrs() == original_attrs

    # Updated space should not have the removed parameter
    updated_attrs = updated_space.get_attrs()
    assert "y" not in updated_attrs
    assert "x" in updated_attrs
    assert "z" in updated_attrs
    assert len(updated_attrs) == len(original_attrs) - 1


def test_remove_method_nonexistent_parameter():
    """Test removing a parameter that doesn't exist."""
    space = BasicSpace()

    with pytest.raises(ValueError, match="No parameter with the name"):
        space.remove("nonexistent")


def test_remove_method_chaining():
    """Test chaining multiple remove operations."""
    space = BasicSpace()

    # Chain multiple removals
    final_space = space.remove("x").remove("y")

    attrs = final_space.get_attrs()
    assert "x" not in attrs
    assert "y" not in attrs
    assert "z" in attrs
    assert len(attrs) == 1


def test_remove_all_parameters():
    """Test removing all parameters from a space."""
    space = BasicSpace()

    # Remove all parameters
    empty_space = space.remove("x").remove("y").remove("z")

    attrs = empty_space.get_attrs()
    assert len(attrs) == 0


# ===== Test add_prior method =====


def test_add_prior_method_basic():
    """Test basic functionality of the add_prior method."""
    space = BasicSpace()
    space.get_attrs()

    # Add prior to a parameter without prior
    updated_space = space.add_prior("x", prior=0.5, prior_confidence=ConfidenceLevel.HIGH)

    # Original space should be unchanged
    original_x = space.get_attrs()["x"]
    assert not original_x.has_prior

    # Updated space should have the prior
    updated_x = updated_space.get_attrs()["x"]
    assert updated_x.has_prior
    assert updated_x.prior == 0.5
    assert updated_x.prior_confidence == ConfidenceLevel.HIGH


def test_add_prior_method_different_types():
    """Test adding priors to different parameter types."""
    space = BasicSpace()

    # Add prior to Float
    space = space.add_prior("x", prior=0.75, prior_confidence=ConfidenceLevel.MEDIUM)
    x_param = space.get_attrs()["x"]
    assert x_param.has_prior
    assert x_param.prior == 0.75

    # Add prior to Integer
    space = space.add_prior("y", prior=7, prior_confidence=ConfidenceLevel.HIGH)
    y_param = space.get_attrs()["y"]
    assert y_param.has_prior
    assert y_param.prior == 7

    # Add prior to Categorical
    space = space.add_prior("z", prior=2, prior_confidence=ConfidenceLevel.LOW)
    z_param = space.get_attrs()["z"]
    assert z_param.has_prior
    assert z_param.prior == 2


def test_add_prior_method_string_confidence():
    """Test add_prior with string confidence levels."""
    space = BasicSpace()

    # Test with string confidence levels
    space = space.add_prior("x", prior=0.3, prior_confidence="low")
    x_param = space.get_attrs()["x"]
    assert x_param.has_prior
    assert x_param.prior == 0.3
    assert x_param.prior_confidence == ConfidenceLevel.LOW

    space = space.add_prior("y", prior=8, prior_confidence="medium")
    y_param = space.get_attrs()["y"]
    assert y_param.prior_confidence == ConfidenceLevel.MEDIUM

    space = space.add_prior("z", prior=0, prior_confidence="high")
    z_param = space.get_attrs()["z"]
    assert z_param.prior_confidence == ConfidenceLevel.HIGH


def test_add_prior_method_nonexistent_parameter():
    """Test adding prior to a parameter that doesn't exist."""
    space = BasicSpace()

    with pytest.raises(ValueError, match="No parameter with the name"):
        space.add_prior("nonexistent", prior=0.5, prior_confidence=ConfidenceLevel.MEDIUM)


def test_add_prior_method_already_has_prior():
    """Test adding prior to a parameter that already has one."""
    space = SpaceWithPriors()

    with pytest.raises(ValueError, match="already has a prior"):
        space.add_prior("x", prior=0.8, prior_confidence=ConfidenceLevel.LOW)


def test_add_prior_method_unsupported_type():
    """Test adding prior to unsupported parameter types."""
    # Create space with an Operation (which doesn't support priors)
    space = BasicSpace()
    op = Operation(operator=lambda x: x * 2, args=(space.get_attrs()["x"],))
    space = space.add(op, "operation_param")

    with pytest.raises(ValueError, match="does not support priors"):
        space.add_prior(
            "operation_param", prior=0.5, prior_confidence=ConfidenceLevel.MEDIUM
        )


# ===== Test combined operations =====


def test_combined_operations():
    """Test combining add, remove, and add_prior operations."""
    space = BasicSpace()

    # Complex chain of operations
    final_space = (
        space.add(Float(lower=50.0, upper=100.0), "new_param")
        .remove("y")
        .add_prior("x", prior=0.25, prior_confidence=ConfidenceLevel.HIGH)
        .add_prior("new_param", prior=75.0, prior_confidence=ConfidenceLevel.MEDIUM)
        .add(Integer(lower=0, upper=10), "another_param")
    )

    attrs = final_space.get_attrs()

    # Check structure
    assert "x" in attrs
    assert "y" not in attrs  # Removed
    assert "z" in attrs
    assert "new_param" in attrs
    assert "another_param" in attrs

    # Check priors
    assert attrs["x"].has_prior
    assert attrs["x"].prior == 0.25
    assert attrs["new_param"].has_prior
    assert attrs["new_param"].prior == 75.0
    assert not attrs["z"].has_prior
    assert not attrs["another_param"].has_prior


def test_immutability():
    """Test that all operations return new instances and don't modify originals."""
    original_space = BasicSpace()
    original_attrs = original_space.get_attrs()

    # Perform various operations
    space1 = original_space.add(Float(lower=0.0, upper=1.0), "temp")
    space2 = original_space.remove("x")
    space3 = original_space.add_prior("y", prior=5, prior_confidence=ConfidenceLevel.HIGH)

    # Original should be unchanged
    assert original_space.get_attrs() == original_attrs
    assert not original_space.get_attrs()["y"].has_prior

    # Each operation should create different instances
    assert space1 is not original_space
    assert space2 is not original_space
    assert space3 is not original_space
    assert space1 is not space2
    assert space2 is not space3


def test_fidelity_operations():
    """Test operations with fidelity parameters."""

    class FidelitySpace(PipelineSpace):
        x = Float(lower=0.0, upper=1.0)
        epochs = Fidelity(Integer(lower=1, upper=100))

    space = FidelitySpace()

    # Add another parameter (non-fidelity since add doesn't support Fidelity directly)
    new_param = Integer(lower=1, upper=50)
    space = space.add(new_param, "batch_size")

    # Check that original fidelity is preserved
    fidelity_attrs = space.fidelity_attrs
    assert "epochs" in fidelity_attrs
    assert len(fidelity_attrs) == 1

    # Remove the fidelity parameter
    space = space.remove("epochs")
    fidelity_attrs = space.fidelity_attrs
    assert "epochs" not in fidelity_attrs
    assert len(fidelity_attrs) == 0

    # Regular parameters should still be there
    regular_attrs = space.get_attrs()
    assert "x" in regular_attrs
    assert "batch_size" in regular_attrs


def test_space_string_representation():
    """Test that string representation works after operations."""
    space = BasicSpace()

    # Perform operations
    modified_space = (
        space.add(Float(lower=10.0, upper=20.0), "added_param")
        .remove("y")
        .add_prior("x", prior=0.8, prior_confidence=ConfidenceLevel.LOW)
    )

    # Should be able to get string representation without error
    str_repr = str(modified_space)
    assert "PipelineSpace" in str_repr
    assert "added_param" in str_repr
    assert "y" not in str_repr  # Should be removed
