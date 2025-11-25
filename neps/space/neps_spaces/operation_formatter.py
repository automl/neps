"""Pretty formatting for Operation objects.

This module provides functionality to convert Operation objects into
human-readable formatted strings. The format is Pythonic and preserves
all information including nested operations, lists, tuples, and dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import Operation


@dataclass
class FormatterStyle:
    """Configuration for the formatting style."""

    indent_str: str = "  "  # Two spaces for indentation
    max_line_length: int = 80  # Try to keep lines under this length
    compact_threshold: int = 40  # Use compact format if repr is shorter
    show_empty_args: bool = True  # Show () for operations with no args/kwargs


def _format_value(
    value: Any,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Format a value (could be primitive, list, tuple, dict, or Operation).

    Args:
        value: The value to format
        indent: Current indentation level
        style: Formatting style configuration

    Returns:
        Formatted string representation of the value
    """
    from neps.space.neps_spaces.parameters import Operation

    if isinstance(value, Operation):
        # Recursively format nested operations
        return _format_operation(value, indent, style)

    if isinstance(value, list | tuple):
        return _format_sequence(value, indent, style)

    if isinstance(value, dict):
        return _format_dict(value, indent, style)

    # For strings that look like identifiers (operation names), don't add quotes
    # to match the previous formatter's behavior
    if isinstance(value, str) and value.isidentifier():
        return value

    # For other primitives, use repr to get proper quoting
    return repr(value)


def _format_sequence(
    seq: list | tuple,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Format a list or tuple, using compact or expanded format as needed."""
    from neps.space.neps_spaces.parameters import Operation

    if not seq:
        return "[]" if isinstance(seq, list) else "()"

    # Try compact format first
    compact = repr(seq)
    if len(compact) <= style.compact_threshold and "\n" not in compact:
        return compact

    # Use expanded format for complex sequences
    bracket_open = "[" if isinstance(seq, list) else "("
    bracket_close = "]" if isinstance(seq, list) else ")"

    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)

    # Check if any element is an Operation (needs expansion)
    has_operations = any(isinstance(item, Operation) for item in seq)

    if has_operations:
        # Full expansion with each item on its own line
        lines = [bracket_open]
        for item in seq:
            formatted = _format_value(item, indent + 1, style)
            lines.append(f"{inner_indent_str}{formatted},")
        lines.append(f"{indent_str}{bracket_close}")
        return "\n".join(lines)

    # Simple items - try to fit multiple per line
    lines = [bracket_open]
    current_line: list[str] = []
    current_length = 0

    for item in seq:
        item_repr = repr(item)
        item_len = len(item_repr) + 2  # +2 for ", "

        if current_line and current_length + item_len > style.max_line_length:
            # Start new line
            lines.append(f"{inner_indent_str}{', '.join(current_line)},")
            current_line = [item_repr]
            current_length = len(item_repr)
        else:
            current_line.append(item_repr)
            current_length += item_len

    if current_line:
        lines.append(f"{inner_indent_str}{', '.join(current_line)},")

    lines.append(f"{indent_str}{bracket_close}")
    return "\n".join(lines)


def _format_dict(
    d: dict,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Format a dictionary."""
    if not d:
        return "{}"

    # Try compact format first
    compact = repr(d)
    if len(compact) <= style.compact_threshold:
        return compact

    # Use expanded format
    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)

    lines = ["{"]
    for key, value in d.items():
        formatted_value = _format_value(value, indent + 1, style)
        lines.append(f"{inner_indent_str}{key!r}: {formatted_value},")
    lines.append(f"{indent_str}}}")
    return "\n".join(lines)


def _format_operation(
    operation: Operation,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Format an Operation object.

    Args:
        operation: The Operation to format
        indent: Current indentation level
        style: Formatting style configuration

    Returns:
        Formatted string representation
    """
    # Get operator name
    operator_name = (
        operation.operator
        if isinstance(operation.operator, str)
        else operation.operator.__name__
    )

    # Check if we have any args or kwargs
    has_args = operation.args and len(operation.args) > 0
    has_kwargs = operation.kwargs and len(operation.kwargs) > 0

    if not has_args and not has_kwargs:
        # Empty operation
        return f"{operator_name}()" if style.show_empty_args else operator_name

    # Always use multi-line format for consistency and readability
    # Build the multi-line formatted string
    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)

    lines = [f"{operator_name}("]

    # Format args
    if has_args:
        for arg in operation.args:
            formatted = _format_value(arg, indent + 1, style)
            lines.append(f"{inner_indent_str}{formatted},")

    # Format kwargs
    if has_kwargs:
        for key, value in operation.kwargs.items():
            formatted_value = _format_value(value, indent + 1, style)
            lines.append(f"{inner_indent_str}{key}={formatted_value},")

    lines.append(f"{indent_str})")

    return "\n".join(lines)


def operation_to_string(
    operation: Operation | Any,
    style: FormatterStyle | None = None,
) -> str:
    """Convert an Operation to a pretty-formatted string.

    This function produces a Pythonic representation of the Operation
    that preserves all information and is easy to read.

    Args:
        operation: The Operation to format (or any value)
        style: Formatting style configuration (uses default if None)

    Returns:
        Pretty-formatted string representation

    Example:
        >>> op = Operation(
        ...     operator=nn.Sequential,
        ...     args=(
        ...         Operation(nn.Conv2d, kwargs={'in_channels': 3, 'kernel_size':
        ...             [3, 3]}),
        ...         Operation(nn.ReLU),
        ...     ),
        ... )
        >>> print(operation_to_string(op))
        Sequential(
          Conv2d(
            in_channels=3,
            kernel_size=[3, 3],
          ),
          ReLU,
        )
    """
    from neps.space.neps_spaces.parameters import Operation

    if style is None:
        style = FormatterStyle()

    if not isinstance(operation, Operation):
        # Not an operation - just format the value
        return _format_value(operation, 0, style)

    return _format_operation(operation, 0, style)


class ConfigString:
    """A class representing a configuration string in NePS spaces.

    This class provides pretty-formatted output for displaying Operation objects
    to users. It's a lightweight wrapper around operation_to_string for backward
    compatibility.
    """

    def __init__(self, config: str | Operation | Any) -> None:
        """Initialize the ConfigString with a configuration.

        Args:
            config: Either a string (for backward compatibility) or an Operation object

        Raises:
            ValueError: If the config is None or empty.
        """
        if config is None or (isinstance(config, str) and len(config) == 0):
            raise ValueError(f"Invalid config: {config}")

        self.config = config

    def pretty_format(self) -> str:
        """Get a pretty formatted string representation of the configuration.

        Returns:
            A Pythonic multi-line string representation with proper indentation.
        """
        from neps.space.neps_spaces.parameters import Operation

        if isinstance(self.config, Operation):
            # Use the formatter for Operation objects
            return operation_to_string(self.config)

        # For string config (backward compatibility), just return as-is
        return str(self.config)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return str(self.config) == str(other.config)
        raise NotImplementedError()  # let the other side check for equality

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return str(self.config).__hash__()
