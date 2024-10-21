"""Constant hyperparameter for search spaces."""

from __future__ import annotations

from typing import Any, TypeVar
from typing_extensions import Self, override

from neps.search_spaces.parameter import Parameter

T = TypeVar("T", int, float, str)


class Constant(Parameter[T, T]):
    """A constant value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with values that should not change during
    optimization. For example, the `batch_size` hyperparameter in a neural
    network search space can be a `Constant` with a value of `32`.

    ```python
    import neps

    batch_size = neps.Constant(32)
    ```

    !!! note

        As the name suggests, the value of a `Constant` only have one
        value and so its [`.default`][neps.search_spaces.parameter.Parameter.default]
        and [`.value`][neps.search_spaces.parameter.Parameter.value] should always be
        the same.

        This also implies that the
        [`.default`][neps.search_spaces.parameter.Parameter.default] can never be `None`.

        Please use
        [`.set_constant_value()`][neps.search_spaces.hyperparameters.constant.Constant.set_constant_value]
        if you need to change the value of the constant parameter.
    """

    def __init__(self, value: T):
        """Create a new `Constant`.

        Args:
            value: value for the hyperparameter.
        """
        super().__init__(value=value, default=value, is_fidelity=False)  # type: ignore
        self._value: T = value  # type: ignore

    @override
    def clone(self) -> Self:
        return self.__class__(value=self.value)

    @property
    @override
    def value(self) -> T:
        """Get the value of the constant parameter."""
        return self._value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value == other.value and self.is_fidelity == other.is_fidelity

    def __repr__(self) -> str:
        return f"<Constant, value: {self.value}>"

    @override
    def sample_value(self) -> T:
        return self.value

    @override
    def set_value(self, value: T | None) -> None:
        """Set the value of the constant parameter.

        !!! note

            This method is a no-op but will raise a `ValueError` if the value
            is different from the current value.

            Please see
            [`.set_constant_value()`][neps.search_spaces.hyperparameters.constant.Constant.set_constant_value]
            which can be used to set both the
            [`.value`][neps.search_spaces.parameter.Parameter.value]
            and the [`.default`][neps.search_spaces.parameter.Parameter.default] at once

        Args:
            value: value to set the parameter to.

        Raises:
            ValueError: if the value is different from the current value.
        """
        if value != self._value:
            raise ValueError(
                f"Constant does not allow chaning the set value. "
                f"Tried to set value to {value}, but it is already {self.value}"
            )

    @override
    def value_to_normalized(self, value: T) -> float:
        return 1.0 if value == self._value else 0.0

    @override
    def normalized_to_value(self, normalized_value: float) -> T:
        return self._value


class ConstantParameter(Constant):
    """Deprecated: Use `Constant` instead of `ConstantParameter`.

    This class remains for backward compatibility and will raise a deprecation
    warning if used.
    """

    def __init__(self, value: T):
        """Initialize a deprecated `ConstantParameter`.

        Args:
            value: value for the hyperparameter.

        Raises:
            DeprecationWarning: A warning indicating that `neps.ConstantParameter` is
            deprecated and `neps.Constant` should be used instead.
        """
        import warnings

        warnings.warn(
            (
                "Usage of 'neps.ConstantParameter' is deprecated and will be removed in"
                " future releases. Please use 'neps.Constant' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(value=value)
