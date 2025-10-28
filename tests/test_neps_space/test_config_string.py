"""Tests for config_string module functions."""

from __future__ import annotations

import pytest

from neps.space.neps_spaces.config_string import (
    ConfigString,
    UnwrappedConfigStringPart,
    unwrap_config_string,
    wrap_config_into_string,
)


class TestUnwrapAndWrapConfigString:
    """Test the unwrap_config_string and wrap_config_into_string functions.

    The new implementation preserves the structure during round-trip unwrap->wrap.
    """

    def test_single_nested_operation(self):
        """Test unwrapping and wrapping a single nested operation."""
        config_str = "Sequential(ReLU)"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        # Round trip should preserve the input
        assert wrapped == config_str

    def test_operation_with_multiple_args(self):
        """Test unwrapping and wrapping an operation with multiple arguments."""
        # New format uses commas: Sequential(ReLU, Conv2D, BatchNorm)
        config_str = "Sequential(ReLU, Conv2D, BatchNorm)"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str

    def test_nested_operations(self):
        """Test unwrapping and wrapping nested operations."""
        # New format: Sequential(Sequential(ReLU), Conv2D)
        config_str = "Sequential(Sequential(ReLU), Conv2D)"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str

    def test_deeply_nested_operations(self):
        """Test unwrapping and wrapping deeply nested operations."""
        # New format: Sequential(Sequential(Sequential(ReLU)))
        config_str = "Sequential(Sequential(Sequential(ReLU)))"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str

    def test_complex_nested_structure(self):
        """Test unwrapping and wrapping a complex nested structure."""
        # New format with multiple levels and operands
        config_str = (
            "Sequential(Sequential(ReLU, Conv2D), BatchNorm, Sequential(Dropout))"
        )
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str

    def test_round_trip_preservation(self):
        """Test that unwrap->wrap round trip preserves the input."""
        test_cases = [
            "Sequential(ReLU)",
            "Sequential(ReLU, Conv2D)",
            "Sequential(Sequential(ReLU), Conv2D)",
            "Sequential(Sequential(Sequential(ReLUConvBN)))",
        ]
        for config_str in test_cases:
            unwrapped = unwrap_config_string(config_str)
            wrapped = wrap_config_into_string(unwrapped)
            assert wrapped == config_str, f"Round trip failed for: {config_str}"

    def test_operation_with_hyperparameters(self):
        """Test that operations with hyperparameters can be unwrapped."""
        # Hyperparameters should be the first element inside the parentheses
        config_str = "Conv2D({kernel_size: 3}, input)"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        # Round trip should preserve the structure
        assert wrapped == config_str

    def test_nested_operation_with_hyperparameters(self):
        """Test unwrapping and wrapping nested operations with hyperparameters."""
        config_str = "Sequential(Conv2D({kernel_size: 3}, input), ReLU)"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        # Should preserve the structure
        assert wrapped == config_str

    def test_resblock_special_case(self):
        """Test the special case handling of 'resBlock resBlock'."""
        # The special handling for resBlock
        config_str = "resBlock resBlock"
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str

    def test_unwrapped_structure(self):
        """Test the structure of unwrapped config parts."""
        config_str = "Sequential(ReLU, Conv2D)"
        unwrapped = unwrap_config_string(config_str)

        # Should have 1 part: Sequential at level 1 with operands "ReLU, Conv2D"
        assert len(unwrapped) == 1

        # First part is Sequential
        assert unwrapped[0].operator == "Sequential"
        assert unwrapped[0].level == 1
        assert unwrapped[0].opening_index == 0
        assert unwrapped[0].operands == "ReLU, Conv2D"

    def test_grammar_like_example(self):
        """Test a realistic example from grammar-like search spaces."""
        # From the actual test expectations
        config_str = (
            "Sequential(Sequential(Sequential(ReLUConvBN)), Sequential(ReLUConvBN),"
            " Identity)"
        )
        unwrapped = unwrap_config_string(config_str)
        wrapped = wrap_config_into_string(unwrapped)
        assert wrapped == config_str


class TestConfigStringClass:
    """Test the ConfigString class."""

    def test_initialization(self):
        """Test ConfigString initialization."""
        config_str = "Sequential(ReLU, Conv2D)"
        cs = ConfigString(config_str)
        assert cs.config_string == config_str

    def test_invalid_initialization(self):
        """Test that ConfigString raises error for invalid input."""
        with pytest.raises(ValueError):
            ConfigString("")
        with pytest.raises((ValueError, TypeError)):
            ConfigString(None)  # type: ignore[arg-type]

    def test_unwrapped_property(self):
        """Test the unwrapped property."""
        config_str = "Sequential(ReLU, Conv2D)"
        cs = ConfigString(config_str)
        unwrapped = cs.unwrapped
        assert isinstance(unwrapped, tuple)
        assert all(isinstance(part, UnwrappedConfigStringPart) for part in unwrapped)

    def test_max_hierarchy_level(self):
        """Test max_hierarchy_level property."""
        config_str = "Sequential(Sequential(Sequential(ReLU)))"
        cs = ConfigString(config_str)
        # Sequential at level 1, Sequential at level 2, Sequential at level 3
        # ReLU doesn't have parentheses so it doesn't create its own unwrapped part
        assert cs.max_hierarchy_level == 3

    def test_at_hierarchy_level(self):
        """Test at_hierarchy_level method."""
        config_str = "Sequential(Sequential(ReLU, Conv2D), BatchNorm)"
        cs = ConfigString(config_str)

        # Get config at level 1 (just the outermost Sequential)
        level_1 = cs.at_hierarchy_level(1)
        assert "Sequential" in level_1.config_string

        # Get config at level 2
        level_2 = cs.at_hierarchy_level(2)
        assert isinstance(level_2, ConfigString)

    def test_equality(self):
        """Test ConfigString equality."""
        cs1 = ConfigString("Sequential(ReLU)")
        cs2 = ConfigString("Sequential(ReLU)")
        cs3 = ConfigString("Sequential(Conv2D)")

        assert cs1 == cs2
        assert cs1 != cs3
        assert cs2 != cs3

    def test_hash(self):
        """Test ConfigString hashing."""
        cs1 = ConfigString("Sequential(ReLU)")
        cs2 = ConfigString("Sequential(ReLU)")

        # Same config strings should have same hash
        assert hash(cs1) == hash(cs2)

        # Should be usable in sets
        config_set = {cs1, cs2}
        assert len(config_set) == 1


class TestOperationSerialization:
    """Test serialization of Operation objects with callables."""

    def test_operation_with_args_and_kwargs(self):
        """Test converting an Operation with args and kwargs to string."""
        from neps.space.neps_spaces.neps_space import convert_operation_to_string
        from neps.space.neps_spaces.parameters import Operation

        # Create an operation with args and kwargs (like Conv2D)
        operation = Operation(
            "Conv2D",
            args=(64,),
            kwargs={"kernel_size": 3, "stride": 1},
        )

        result = convert_operation_to_string(operation)
        # Should serialize to Conv2D(64)
        assert "Conv2D" in result
        assert "64" in result

    def test_nested_operations_with_multiple_args(self):
        """Test converting nested operations with multiple args."""
        from neps.space.neps_spaces.neps_space import convert_operation_to_string
        from neps.space.neps_spaces.parameters import Operation

        # Create nested operations like Sequential(ReLU, Conv2D(64))
        conv_op = Operation("Conv2D", args=(64,), kwargs={"kernel_size": 3})
        relu_op = Operation("ReLU", args=(), kwargs={})
        sequential_op = Operation("Sequential", args=(relu_op, conv_op), kwargs={})

        result = convert_operation_to_string(sequential_op)
        # Should contain all operators
        assert "Sequential" in result
        assert "ReLU" in result
        assert "Conv2D" in result
        assert "64" in result

    def test_operation_with_callable_operator(self):
        """Test converting an Operation with a callable operator."""
        from neps.space.neps_spaces.neps_space import convert_operation_to_string
        from neps.space.neps_spaces.parameters import Operation

        # Define a simple callable
        def my_layer(in_features, out_features):
            return f"MyLayer({in_features}, {out_features})"

        # Create operation with callable
        operation = Operation(my_layer, args=(128, 64), kwargs={})

        result = convert_operation_to_string(operation)
        # Should use the callable's name
        assert "my_layer" in result

    def test_operation_serialization_with_mixed_args(self):
        """Test operation with mix of simple args and operations."""
        from neps.space.neps_spaces.neps_space import convert_operation_to_string
        from neps.space.neps_spaces.parameters import Operation

        # Create nested operation with mixed types
        inner = Operation("ReLU", args=(), kwargs={})
        outer = Operation("Sequential", args=(inner, "BatchNorm"), kwargs={})

        result = convert_operation_to_string(outer)
        # Should serialize both operations and simple strings
        assert "Sequential" in result
        assert "ReLU" in result
        assert "BatchNorm" in result

    def test_round_trip_with_operations(self):
        """Test that operations can round-trip through unwrap/wrap."""
        from neps.space.neps_spaces.neps_space import convert_operation_to_string
        from neps.space.neps_spaces.parameters import Operation

        # Create a complex nested structure
        conv1 = Operation("Conv2D", args=(32,), kwargs={"kernel_size": 3})
        relu = Operation("ReLU", args=(), kwargs={})
        conv2 = Operation("Conv2D", args=(64,), kwargs={"kernel_size": 3})
        sequential = Operation("Sequential", args=(conv1, relu, conv2), kwargs={})

        # Convert to string
        config_str = convert_operation_to_string(sequential)

        # Verify it's parseable (unwrap should work)
        unwrapped = unwrap_config_string(config_str)
        assert len(unwrapped) > 0

        # Verify it can be wrapped back
        rewrapped = wrap_config_into_string(tuple(unwrapped))
        assert rewrapped == config_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
