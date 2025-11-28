"""Comprehensive tests for string_formatter module."""

from __future__ import annotations

import neps
from neps.space.neps_spaces.parameters import Operation
from neps.space.neps_spaces.string_formatter import (
    FormatterStyle,
    operation_to_string,
)


def test_simple_operation_no_args():
    """Test formatting an operation with no arguments - default shows ()."""
    op = Operation(operator="ReLU")
    result = operation_to_string(op)
    assert result == "ReLU()"


def test_simple_operation_no_args_with_parens():
    """Test formatting with show_empty_args=False to hide ()."""
    op = Operation(operator="ReLU")
    style = FormatterStyle(show_empty_args=False)
    result = operation_to_string(op, style)
    assert result == "ReLU"


def test_operation_with_args_only():
    """Test formatting an operation with positional args only - always expanded."""
    op = Operation(operator="Add", args=(1, 2, 3))
    result = operation_to_string(op)
    expected = """Add(
   1,
   2,
   3,
)"""
    assert result == expected


def test_operation_with_kwargs_only():
    """Test formatting an operation with keyword args only - always expanded."""
    op = Operation(operator="Conv2d", kwargs={"in_channels": 3, "out_channels": 64})
    result = operation_to_string(op)
    expected = """Conv2d(
   in_channels=3,
   out_channels=64,
)"""
    assert result == expected


def test_operation_with_args_and_kwargs():
    """Test formatting with both positional and keyword arguments - always expanded."""
    op = Operation(
        operator="LinearLayer",
        args=(128,),
        kwargs={"activation": "relu", "dropout": 0.5},
    )
    result = operation_to_string(op)
    expected = """LinearLayer(
   128,
   activation=relu,
   dropout=0.5,
)"""
    assert result == expected


def test_nested_operations():
    """Test formatting nested operations."""
    inner = Operation(operator="ReLU")
    outer = Operation(operator="Sequential", args=(inner,))
    result = operation_to_string(outer)
    expected = """Sequential(
   ReLU(),
)"""
    assert result == expected


def test_deeply_nested_operations():
    """Test formatting deeply nested operations - all ops expanded."""
    conv = Operation(
        operator="Conv2d",
        kwargs={"in_channels": 3, "out_channels": 64, "kernel_size": 3},
    )
    relu = Operation(operator="ReLU")
    pool = Operation(operator="MaxPool2d", kwargs={"kernel_size": 2})

    sequential = Operation(operator="Sequential", args=(conv, relu, pool))

    result = operation_to_string(sequential)
    expected = """Sequential(
   Conv2d(
      in_channels=3,
      out_channels=64,
      kernel_size=3,
   ),
   ReLU(),
   MaxPool2d(
      kernel_size=2,
   ),
)"""
    assert result == expected


def test_list_as_arg():
    """Test formatting with a list as an argument."""
    op = Operation(operator="Conv2d", kwargs={"kernel_size": [3, 3]})
    result = operation_to_string(op)
    expected = """Conv2d(
   kernel_size=[3, 3],
)"""
    assert result == expected


def test_long_list_as_arg():
    """Test formatting with a longer list that spans multiple lines."""
    long_list = list(range(20))
    op = Operation(operator="SomeOp", kwargs={"values": long_list})
    result = operation_to_string(op)

    # Should have the list expanded
    assert "values=[" in result
    assert "0, 1, 2" in result  # Multiple items per line
    assert "]" in result


def test_tuple_as_arg():
    """Test formatting with a tuple as an argument."""
    op = Operation(operator="Shape", args=((64, 64, 3),))
    result = operation_to_string(op)
    expected = """Shape(
   (64, 64, 3),
)"""
    assert result == expected


def test_dict_as_kwarg():
    """Test formatting with a dict as a keyword argument value."""
    op = Operation(
        operator="ConfigOp",
        kwargs={"config": {"learning_rate": 0.001, "batch_size": 32}},
    )
    result = operation_to_string(op)
    # Dict gets expanded due to length
    expected = """ConfigOp(
   config={
      'learning_rate': 0.001,
      'batch_size': 32,
   },
)"""
    assert result == expected


def test_operations_in_list():
    """Test formatting operations inside a list argument - all ops expanded."""
    op1 = Operation(operator="Conv2d", kwargs={"channels": 32})
    op2 = Operation(operator="Conv2d", kwargs={"channels": 64})

    container = Operation(operator="ModuleList", args=([op1, op2],))

    result = operation_to_string(container)
    expected = """ModuleList(
   [
      Conv2d(
         channels=32,
      ),
      Conv2d(
         channels=64,
      ),
   ],
)"""
    assert result == expected


def test_operations_in_list_as_kwarg():
    """Test formatting operations inside a list that is a kwarg value."""
    op1 = Operation(operator="ReLU")
    op2 = Operation(operator="Sigmoid")

    container = Operation(operator="Container", kwargs={"layers": [op1, op2]})

    result = operation_to_string(container)
    expected = """Container(
   layers=[
      ReLU(),
      Sigmoid(),
   ],
)"""
    assert result == expected


def test_mixed_types_in_list():
    """Test formatting a list with mixed types including operations."""
    op = Operation(operator="ReLU")
    mixed_list = [1, "hello", 3.14, op, [1, 2, 3]]

    container = Operation(operator="MixedContainer", args=(mixed_list,))

    result = operation_to_string(container)

    # Check that all elements are present
    assert "1," in result
    assert "hello," in result  # Identifiers don't get quotes
    assert "3.14," in result
    assert "ReLU()," in result
    assert "[1, 2, 3]," in result


def test_string_values_with_quotes():
    """Test that string values are properly quoted."""
    op = Operation(
        operator="TextOp",
        kwargs={
            "text": "hello world",
            "quote_test": "it's a test",
            "double_quotes": 'say "hello"',
        },
    )
    result = operation_to_string(op)

    # Check strings are properly represented
    assert "text='hello world'" in result or 'text="hello world"' in result
    assert "quote_test" in result
    assert "double_quotes" in result


def test_complex_nested_structure():
    """Test a complex nested structure with all types."""
    # Build a complex structure

    conv = Operation(
        operator="Conv2d",
        kwargs={"in_channels": 3, "out_channels": 64, "kernel_size": [3, 3]},
    )
    relu = Operation(operator="ReLU")

    seq = Operation(
        operator="Sequential",
        args=([conv, relu],),
        kwargs={"dropout": 0.5, "config": {"layers": [3, 64, 128]}},
    )

    result = operation_to_string(seq)

    # Verify structure
    assert "Sequential(" in result
    assert "Conv2d(" in result
    assert "in_channels=3" in result
    assert "kernel_size=[3, 3]" in result
    assert "ReLU()," in result
    assert "dropout=0.5" in result
    assert "config=" in result
    assert "'layers': [3, 64, 128]" in result


def test_non_operation_value():
    """Test formatting a non-Operation value."""
    # Should work with any value
    result1 = operation_to_string(42)
    assert result1 == "42"

    result2 = operation_to_string("hello")
    assert result2 == "hello"  # Identifiers don't get quotes

    result3 = operation_to_string([1, 2, 3])
    assert result3 == "[1, 2, 3]"


def test_custom_indent():
    """Test using a custom indentation style - all ops expanded."""
    op = Operation(operator="Conv2d", kwargs={"channels": 64})
    style = FormatterStyle(indent_str="    ")  # 4 spaces

    result = operation_to_string(op, style)
    expected = """Conv2d(
    channels=64,
)"""
    assert result == expected


def test_empty_list():
    """Test formatting with empty list."""
    op = Operation(operator="Op", kwargs={"items": []})
    result = operation_to_string(op)
    expected = """Op(
   items=[],
)"""
    assert result == expected


def test_empty_tuple():
    """Test formatting with empty tuple."""
    op = Operation(operator="Op", args=((),))
    result = operation_to_string(op)
    expected = """Op(
   (),
)"""
    assert result == expected


def test_empty_dict():
    """Test formatting with empty dict."""
    op = Operation(operator="Op", kwargs={"config": {}})
    result = operation_to_string(op)
    expected = """Op(
   config={},
)"""
    assert result == expected


def test_boolean_values():
    """Test formatting with boolean values - always expanded."""
    op = Operation(operator="Op", kwargs={"enabled": True, "debug": False, "count": 0})
    result = operation_to_string(op)
    expected = """Op(
   enabled=True,
   debug=False,
   count=0,
)"""
    assert result == expected


def test_none_value():
    """Test formatting with None value - always expanded."""
    op = Operation(operator="Op", kwargs={"default": None})
    result = operation_to_string(op)
    expected = """Op(
   default=None,
)"""
    assert result == expected


def test_real_world_example():
    """Test a realistic neural network architecture."""
    # Build a realistic example similar to architecture_search.py
    conv1 = Operation(
        operator="Conv2d",
        kwargs={"in_channels": 3, "out_channels": 64, "kernel_size": [3, 3]},
    )
    relu1 = Operation(operator="ReLU")
    pool1 = Operation(operator="MaxPool2d", kwargs={"kernel_size": 2, "stride": 2})

    conv2 = Operation(
        operator="Conv2d",
        kwargs={"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3]},
    )
    relu2 = Operation(operator="ReLU")
    pool2 = Operation(operator="MaxPool2d", kwargs={"kernel_size": 2, "stride": 2})

    flatten = Operation(operator="Flatten")
    fc = Operation(
        operator="Linear", kwargs={"in_features": 128 * 7 * 7, "out_features": 10}
    )

    model = Operation(
        operator="Sequential",
        args=([conv1, relu1, pool1, conv2, relu2, pool2, flatten, fc],),
    )

    result = operation_to_string(model)

    # Verify key elements are present
    assert "Sequential(" in result
    assert "Conv2d(" in result
    assert "in_channels=3" in result
    assert "out_channels=64" in result
    assert "kernel_size=[3, 3]" in result
    assert "ReLU()," in result
    assert "MaxPool2d(" in result
    assert "Flatten()," in result
    assert "Linear(" in result
    assert "in_features=" in result
    assert "out_features=10" in result


def test_categorical_with_operations():
    """Test formatting when a Categorical contains Operations - always expanded."""

    class TestSpace(neps.PipelineSpace):
        choice = neps.Categorical(
            [
                Operation(operator="Conv2d", kwargs={"in_channels": 3, "kernel_size": 3}),
                Operation(operator="ReLU"),
            ]
        )

    # Sample and resolve
    space = TestSpace()
    resolved, _ = neps.space.neps_spaces.neps_space.resolve(space)

    # The resolved choice should be an Operation
    assert isinstance(resolved.choice, Operation)

    # Should format properly - check for essential content
    result = operation_to_string(resolved.choice)
    # Either Conv2d with both params, or ReLU
    is_conv = (
        "Conv2d" in result and "in_channels=3" in result and "kernel_size=3" in result
    )
    is_relu = result == "ReLU()"
    assert is_conv or is_relu


def test_categorical_with_primitives():
    """Test formatting when a Categorical contains primitives."""

    class TestSpace(neps.PipelineSpace):
        choice = neps.Categorical(["adam", "sgd", "rmsprop"])

    space = TestSpace()
    resolved, _ = neps.space.neps_spaces.neps_space.resolve(space)

    # The resolved choice should be a string
    assert isinstance(resolved.choice, str)

    # Should format as a simple string (identifiers don't get quotes)
    result = operation_to_string(resolved.choice)
    assert result in ["adam", "sgd", "rmsprop"]


def test_categorical_with_mixed_types():
    """Test formatting when a Categorical contains mixed types."""

    class TestSpace(neps.PipelineSpace):
        choice = neps.Categorical(
            [
                Operation(operator="Linear", kwargs={"in_features": 10}),
                "simple_string",
                42,
            ]
        )

    space = TestSpace()
    resolved, _ = neps.space.neps_spaces.neps_space.resolve(space)

    # Should format appropriately based on what was chosen
    result = operation_to_string(resolved.choice)

    # Check it's one of the expected formats (identifiers don't get quotes)
    possible_results = [
        "Linear(\n  in_features=10,\n)",  # Expanded format
        "Linear(in_features=10)",  # Compact format (simple operation)
        "simple_string",  # Identifiers don't get quotes
        "42",
    ]
    assert result in possible_results


def test_resolved_float_parameter():
    """Test formatting a resolved Float parameter."""

    class TestSpace(neps.PipelineSpace):
        lr = neps.Float(0.001, 0.1)

    space = TestSpace()
    resolved, _ = neps.space.neps_spaces.neps_space.resolve(space)

    # Resolved Float becomes a float value
    assert isinstance(resolved.lr, float)

    # Should format as a simple number
    result = operation_to_string(resolved.lr)
    assert result == repr(resolved.lr)


def test_resolved_integer_parameter():
    """Test formatting a resolved Integer parameter."""

    class TestSpace(neps.PipelineSpace):
        batch_size = neps.Integer(16, 128)

    space = TestSpace()
    resolved, _ = neps.space.neps_spaces.neps_space.resolve(space)

    # Resolved Integer becomes an int value
    assert isinstance(resolved.batch_size, int)

    # Should format as a simple number
    result = operation_to_string(resolved.batch_size)
    assert result == repr(resolved.batch_size)


if __name__ == "__main__":
    # Run a quick test to see output
    conv = Operation(
        operator="Conv2d",
        kwargs={"in_channels": 3, "out_channels": 64, "kernel_size": [3, 3]},
    )
    relu = Operation(operator="ReLU")
    seq = Operation(operator="Sequential", args=([conv, relu],), kwargs={"dropout": 0.5})

    import pytest

    pytest.main([__file__, "-v"])
