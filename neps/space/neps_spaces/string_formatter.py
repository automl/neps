"""Pretty formatting for Operation objects.

This module provides functionality to convert Operation objects into
human-readable formatted strings. The format is Pythonic and preserves
all information including nested operations, lists, tuples, and dicts.

ARCHITECTURE:
    format_value() - Single entry point for ALL formatting
        ├── _format_categorical() - Internal handler for Categorical
        ├── _format_float() - Internal handler for Float
        ├── _format_integer() - Internal handler for Integer
        ├── _format_resampled() - Internal handler for Resample
        ├── _format_repeated() - Internal handler for Repeated
        ├── _format_operation() - Internal handler for Operation
        ├── _format_sequence() - Internal handler for list/tuple
        └── _format_dict() - Internal handler for dict

All __str__ methods should call format_value() directly.
All internal formatters call format_value() for nested values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import (
        Categorical,
        Float,
        Integer,
        Operation,
        Repeated,
        Resample,
    )


@dataclass
class FormatterStyle:
    """Configuration for the formatting style."""

    indent_str: str = "   "  # Three spaces for indentation
    max_line_length: int = 90  # Try to keep lines under this length
    compact_threshold: int = 40  # Use compact format if repr is shorter
    show_empty_args: bool = True  # Show () for operations with no args/kwargs


# ============================================================================
# PUBLIC API - Single entry point for all formatting
# ============================================================================


def format_value(  # noqa: C901, PLR0911, PLR0912
    value: Any,
    indent: int = 0,
    style: FormatterStyle | None = None,
) -> str:
    """Format any value with proper indentation and style.

    This is the SINGLE entry point for all formatting in NePS.
    All __str__ methods should delegate to this function.

    Args:
        value: The value to format (any type)
        indent: Current indentation level
        style: Formatting style configuration

    Returns:
        Formatted string representation
    """
    from neps.space.neps_spaces.parameters import (
        Categorical,
        Fidelity,
        Float,
        Integer,
        Operation,
        Repeated,
        Resample,
    )

    if style is None:
        style = FormatterStyle()

    # Dispatch to appropriate internal formatter based on type
    if isinstance(value, Operation):
        return _format_operation(value, indent, style)

    if isinstance(value, Categorical):
        return _format_categorical(value, indent, style)

    if isinstance(value, Float):
        return _format_float(value, indent, style)

    if isinstance(value, Integer):
        return _format_integer(value, indent, style)

    if isinstance(value, Fidelity):
        # Use the __str__ method of Fidelity subclasses directly
        return str(value)

    if isinstance(value, Resample):
        return _format_resampled(value, indent, style)

    if isinstance(value, Repeated):
        return _format_repeated(value, indent, style)

    if isinstance(value, list | tuple):
        return _format_sequence(value, indent, style)

    if isinstance(value, dict):
        return _format_dict(value, indent, style)

    # Check for PipelineSpace (import here to avoid circular dependency)
    from neps.space.neps_spaces.parameters import PipelineSpace

    if isinstance(value, PipelineSpace):
        return _format_pipeline_space(value, indent, style)

    # For callables (functions, methods), show their name
    if callable(value) and (name := getattr(value, "__name__", None)):
        return name

    # For identifier strings, don't add quotes
    if isinstance(value, str) and value.isidentifier():
        return value

    # For other values, use repr
    return repr(value)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _collapse_closing_brackets(text: str) -> str:
    """Collapse consecutive closing brackets onto same line respecting indentation.

    Transforms:
           )
           )
        )
    Into:
        ) ) )

    All brackets are placed on the same line using the minimum indentation.

    Args:
        text: The formatted text

    Returns:
        Text with collapsed closing brackets
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        current_line = lines[i]
        stripped = current_line.strip()

        # Check if this line contains only closing brackets
        if stripped and all(c in ")]" for c in stripped):
            # Collect consecutive bracket lines
            bracket_lines = [current_line]
            j = i + 1
            while (
                j < len(lines)
                and lines[j].strip()
                and all(c in ")]" for c in lines[j].strip())
            ):
                bracket_lines.append(lines[j])
                j += 1

            # Collapse if multiple bracket lines
            if len(bracket_lines) > 1:
                # Find minimum indentation
                min_indent = min(len(line) - len(line.lstrip()) for line in bracket_lines)
                # Collapse onto single line
                combined = " ".join(line.strip() for line in bracket_lines)
                result.append(" " * min_indent + combined)
            else:
                result.append(current_line)

            i = j
        else:
            result.append(current_line)
            i += 1

    return "\n".join(result)


# ============================================================================
# INTERNAL FORMATTERS - Type-specific formatting logic
# All these call format_value() for nested values to maintain consistency
# ============================================================================


def _format_prior_confidence(prior_confidence: Any) -> str:
    """Internal helper to format prior_confidence values consistently.

    Args:
        prior_confidence: The prior confidence value (typically a ConfidenceLevel enum)

    Returns:
        String representation of the prior confidence
    """
    return (
        prior_confidence.value
        if hasattr(prior_confidence, "value")
        else str(prior_confidence)
    )


def _format_categorical(
    categorical: Categorical,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for Categorical parameters."""
    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)
    choice_indent_str = style.indent_str * (indent + 2)

    # Format each choice using format_value for consistency
    formatted_choices = []
    for choice in categorical.choices:  # type: ignore[union-attr]
        choice_str = format_value(choice, indent + 2, style)
        formatted_choices.append(choice_str)

    # Check if all choices are simple (strings or numbers without newlines)
    all_simple = all("\n" not in choice_str for choice_str in formatted_choices)

    if all_simple and formatted_choices:
        # Try to fit choices on one line
        choices_str = ", ".join(formatted_choices)
        if len(choices_str) <= style.max_line_length:
            # Put choices on own line, indented
            result = f"Categorical(\n{inner_indent_str}choices=({choices_str})"
        else:
            # Put on multiple lines but keep choices together
            choices_str = f",\n{choice_indent_str}".join(formatted_choices)
            result = (
                f"Categorical(\n{inner_indent_str}choices=(\n"
                f"{choice_indent_str}{choices_str})"
            )
    else:
        # Complex choices - use multi-line format
        choices_str = f",\n{choice_indent_str}".join(formatted_choices)
        result = (
            f"Categorical(\n{inner_indent_str}choices=(\n"
            f"{choice_indent_str}{choices_str}\n{inner_indent_str})"
        )

    if categorical.has_prior:
        prior_confidence_str = _format_prior_confidence(categorical._prior_confidence)
        result += (
            f",\n{inner_indent_str}prior={categorical._prior},"
            f"\n{inner_indent_str}prior_confidence={prior_confidence_str}"
        )

    result += f"\n{indent_str})"
    return _collapse_closing_brackets(result)


def _format_float(
    float_param: Float,
    indent: int,  # noqa: ARG001
    style: FormatterStyle,  # noqa: ARG001
) -> str:
    """Internal formatter for Float parameters."""
    string = f"Float({float_param._lower}, {float_param._upper}"
    if float_param._log:
        string += ", log"
    if float_param.has_prior:
        prior_confidence_str = _format_prior_confidence(float_param._prior_confidence)
        string += f", prior={float_param._prior}, prior_confidence={prior_confidence_str}"
    string += ")"
    return string


def _format_integer(
    integer_param: Integer,
    indent: int,  # noqa: ARG001
    style: FormatterStyle,  # noqa: ARG001
) -> str:
    """Internal formatter for Integer parameters."""
    string = f"Integer({integer_param._lower}, {integer_param._upper}"
    if integer_param._log:
        string += ", log"
    if integer_param.has_prior:
        prior_confidence_str = _format_prior_confidence(integer_param._prior_confidence)
        string += (
            f", prior={integer_param._prior}, prior_confidence={prior_confidence_str}"
        )
    string += ")"
    return string


def _format_resampled(
    resampled: Resample,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for Resample parameters."""
    source = resampled._source

    # Format the source using unified format_value
    source_str = format_value(source, indent + 1, style)

    # Use multi-line format if source is multi-line
    if "\n" in source_str:
        indent_str = style.indent_str * indent
        inner_indent_str = style.indent_str * (indent + 1)
        result = f"Resample(\n{inner_indent_str}{source_str}\n{indent_str})"
        return _collapse_closing_brackets(result)

    # Simple single-line format for basic types
    return f"Resample({source_str})"


def _format_repeated(
    repeated: Repeated,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for Repeated parameters."""
    source_str = format_value(repeated._content, indent, style)
    return f"Repeated({source_str})"


def _format_sequence(
    seq: list | tuple,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for lists and tuples."""
    from neps.space.neps_spaces.parameters import Operation

    if not seq:
        return "[]" if isinstance(seq, list) else "()"

    # Format all items
    formatted_items = [format_value(item, indent + 1, style) for item in seq]

    # Check for "Nx" shorthand case (all items identical)
    if len(set(formatted_items)) == 1 and len(seq) > 1:
        return f"{len(seq)}x {formatted_items[0]}"

    # Try compact format for simple sequences
    compact = repr(seq)
    if len(compact) <= style.compact_threshold and "\n" not in compact:
        return compact

    # Expand multi-line or complex sequences
    is_list = isinstance(seq, list)
    bracket_open, bracket_close = ("[", "]") if is_list else ("(", ")")
    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)

    # Check if expansion is needed (Operations or multi-line items)
    needs_expansion = any(
        isinstance(item, Operation) or "\n" in item_str
        for item, item_str in zip(seq, formatted_items, strict=False)
    )

    if needs_expansion:
        # Full expansion: each item on its own line
        lines = [bracket_open]
        lines.extend(f"{inner_indent_str}{item}," for item in formatted_items)
        lines.append(f"{indent_str}{bracket_close}")
    else:
        # Compact expansion: fit multiple items per line
        lines = [bracket_open]
        current_line: list[str] = []
        current_length = 0

        for item_str in formatted_items:
            item_len = len(item_str) + 2  # +2 for ", "
            if current_line and current_length + item_len > style.max_line_length:
                lines.append(f"{inner_indent_str}{', '.join(current_line)},")
                current_line, current_length = [item_str], len(item_str)
            else:
                current_line.append(item_str)
                current_length += item_len

        if current_line:
            lines.append(f"{inner_indent_str}{', '.join(current_line)},")
        lines.append(f"{indent_str}{bracket_close}")

    result = "\n".join(lines)
    return _collapse_closing_brackets(result)


def _format_dict(
    d: dict,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for dictionaries."""
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
        formatted_value = format_value(value, indent + 1, style)
        lines.append(f"{inner_indent_str}{key!r}: {formatted_value},")
    lines.append(f"{indent_str}}}")
    return "\n".join(lines)


def _format_operation(
    operation: Operation,
    indent: int,
    style: FormatterStyle,
) -> str:
    """Internal formatter for Operation objects."""
    # Get operator name
    operator_name = (
        operation.operator
        if isinstance(operation.operator, str)
        else operation.operator.__name__
    )

    # Check if we have any args or kwargs
    has_args = bool(operation.args)
    has_kwargs = bool(operation.kwargs)

    if not (has_args or has_kwargs):
        return f"{operator_name}()" if style.show_empty_args else operator_name

    # Always use multi-line format for consistency
    indent_str = style.indent_str * indent
    inner_indent_str = style.indent_str * (indent + 1)

    lines = [f"{operator_name}("]

    # Format args using format_value
    if has_args:
        for arg in operation.args:
            formatted = format_value(arg, indent + 1, style)
            lines.append(f"{inner_indent_str}{formatted},")

    # Format kwargs using format_value
    if has_kwargs:
        for key, value in operation.kwargs.items():
            formatted_value = format_value(value, indent + 1, style)
            lines.append(f"{inner_indent_str}{key}={formatted_value},")

    lines.append(f"{indent_str})")

    return "\n".join(lines)


def _format_pipeline_space(
    pipeline_space: Any,
    indent: int,  # noqa: ARG001
    style: FormatterStyle,
) -> str:
    """Internal formatter for PipelineSpace objects."""
    lines = [f"{pipeline_space.__class__.__name__} with parameters:"]
    for k, v in pipeline_space.get_attrs().items():
        if not k.startswith("_") and not callable(v):
            # Use the unified formatter for all values
            formatted_value = format_value(v, 0, style)
            # If multi-line, indent all lines
            if "\n" in formatted_value:
                indented_value = "\n    ".join(formatted_value.split("\n"))
                lines.append(f"  {k}:\n    {indented_value}")
            else:
                lines.append(f"  {k} = {formatted_value}")
    return "\n".join(lines)
