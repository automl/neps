"""This module provides functionality to format configuration strings
used in NePS spaces for display purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neps.space.neps_spaces.operation_formatter import operation_to_string

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import Operation


class ConfigString:
    """A class representing a configuration string in NePS spaces.

    This class provides pretty-formatted output for displaying Operation objects
    to users. It uses the new operation_formatter module internally.
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
            # Use the new formatter for Operation objects
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
