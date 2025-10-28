"""This module defines various classes and protocols for representing and manipulating
search spaces in NePS (Neural Parameter Search). It includes definitions for domains,
pipelines, operations, and fidelity, as well as utilities for sampling and resolving
search spaces.
"""

from __future__ import annotations

import abc
import enum
import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

T = TypeVar("T")


class _Unset:
    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _Unset()


def _parameters_are_equivalent(param1: Any, param2: Any) -> bool:
    """Check if two parameters are equivalent using their is_equivalent_to method.

    This helper function provides a safe way to compare parameters without
    interfering with Python's object identity system. Falls back to regular
    equality comparison for objects that don't have the is_equivalent_to method.

    Args:
        param1: First parameter to compare.
        param2: Second parameter to compare.

    Returns:
        True if the parameters are equivalent, False otherwise.
    """
    # Try to use the is_equivalent_to method if available
    if hasattr(param1, "is_equivalent_to"):
        return param1.is_equivalent_to(param2)
    if hasattr(param2, "is_equivalent_to"):
        return param2.is_equivalent_to(param1)
    # Fall back to regular equality for other types
    return param1 == param2


@runtime_checkable
class Resolvable(Protocol):
    """A protocol for objects that can be resolved into attributes."""

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the resolvable object as a mapping."""
        raise NotImplementedError()

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        """Create a new resolvable object from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new resolvable object with the specified attributes.
        """
        raise NotImplementedError()


def resolvable_is_fully_resolved(resolvable: Resolvable) -> bool:
    """Check if a resolvable object is fully resolved.
    A resolvable object is considered fully resolved if all its attributes are either
    not instances of Resolvable or are themselves fully resolved.

    Args:
      resolvable: Resolvable:

    Returns:
        bool: True if the resolvable object is fully resolved, False otherwise.
    """
    attr_objects = resolvable.get_attrs().values()
    return all(
        not isinstance(obj, Resolvable) or resolvable_is_fully_resolved(obj)
        for obj in attr_objects
    )


class Fidelity(Resolvable, Generic[T]):
    """A class representing a fidelity in a NePS space.

    Attributes:
        domain: The domain of the fidelity, which can be an Integer or Float domain.
    """

    def __init__(self, domain: Integer | Float):
        """Initialize the Fidelity with a domain.

        Args:
            domain: The domain of the fidelity, which can be an Integer or Float domain.

        """
        if domain.has_prior:
            raise ValueError(
                "The domain of a Fidelity can not have priors, has prior:"
                f" {domain.prior!r}."
            )
        self._domain = domain

    def __str__(self) -> str:
        """Get a string representation of the fidelity."""
        return f"Fidelity({self._domain.__str__()})"

    def compare_domain_to(self, other: object) -> bool:
        """Check if this fidelity parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Fidelity):
            return False
        return self._domain == other._domain

    @property
    def min_value(self) -> int | float:
        """Get the minimum value of the fidelity domain.

        Returns:
            The minimum value of the fidelity domain.
        """
        return self._domain.min_value

    @property
    def max_value(self) -> int | float:
        """Get the maximum value of the fidelity domain.

        Returns:
            The maximum value of the fidelity domain.
        """
        return self._domain.max_value

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the fidelity as a mapping.
        This method collects all attributes of the fidelity class and instance,
        excluding private attributes and methods, and returns them as a dictionary.

        Returns:
          A mapping of attribute names to their values.

        Raises:
          ValueError: If the fidelity has no domain defined.

        """
        raise ValueError("For a Fidelity object there is nothing to resolve.")

    def from_attrs(self, attrs: Mapping[str, Any]) -> Fidelity:  # noqa: ARG002
        """Create a new Fidelity instance from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new Fidelity instance with the specified attributes.

        Raises:
            ValueError: If the fidelity has no domain defined.

        """
        raise ValueError("For a Fidelity object there is nothing to resolve.")


class PipelineSpace(Resolvable):
    """A class representing a pipeline in NePS spaces."""

    @property
    def fidelity_attrs(self) -> Mapping[str, Fidelity]:
        """Get the fidelity attributes of the pipeline. Fidelity attributes are special
        attributes that represent the fidelity of the pipeline.

        Returns:
            A mapping of attribute names to Fidelity objects.
        """
        return {k: v for k, v in self.get_attrs().items() if isinstance(v, Fidelity)}

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the pipeline as a mapping.
        This method collects all attributes of the pipeline class and instance,
        excluding private attributes and methods, and returns them as a dictionary.

        Returns:
            A mapping of attribute names to their values.
        """
        attrs = {}

        for attr_name, attr_value in vars(self.__class__).items():
            if attr_name.startswith("_") or callable(attr_value) or attr_value is None:
                continue
            attrs[attr_name] = attr_value

        for attr_name, attr_value in vars(self).items():
            if attr_name.startswith("_") or callable(attr_value) or attr_value is None:
                continue
            attrs[attr_name] = attr_value

        properties_to_ignore = ("fidelity_attrs",)
        for property_to_ignore in properties_to_ignore:
            attrs.pop(property_to_ignore, None)

        return attrs

    def from_attrs(self, attrs: Mapping[str, Any]) -> PipelineSpace:
        """Create a new Pipeline instance from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.


        Returns:
            A new Pipeline instance with the specified attributes.

        Raises:
            ValueError: If the attributes do not match the pipeline's expected structure.
        """
        new_pipeline = PipelineSpace()
        for name, value in attrs.items():
            setattr(new_pipeline, name, value)
        return new_pipeline

    def __str__(self) -> str:
        """Get a string representation of the pipeline.

        Returns:
            A string representation of the pipeline, including its class name and
            attributes.
        """
        attrs = "\n\t".join(
            f"{k} = {v!s}"
            for k, v in self.get_attrs().items()
            if not k.startswith("_") and not callable(v)
        )
        return f"PipelineSpace {self.__class__.__name__} with parameters:\n\t{attrs}"

    def add(
        self,
        new_param: Integer | Float | Categorical | Operation | Resampled | Repeated,
        name: str | None = None,
    ) -> PipelineSpace:
        """Add a new parameter to the pipeline.

        Args:
            new_param: The parameter to be added, which can be an Integer, Float,
                Categorical, Operation, Resampled, Repeated, or PipelineSpace.
            name: The name of the parameter to be added. If None, a default name will be
                generated.

        Returns:
            A new PipelineSpace instance with the added parameter.

        Raises:
            ValueError: If the parameter is not of a supported type or if a parameter
                with the same name already exists in the pipeline.
        """
        if isinstance(new_param, PipelineSpace):
            new_space = self
            for exist_name, value in new_param.get_attrs().items():
                new_space = new_space.add(value, exist_name)
            return new_space

        if not isinstance(
            new_param, Integer | Float | Categorical | Operation | Resampled | Repeated
        ):
            raise ValueError(
                "Can only add Integer, Float, Categorical, Operation, Resampled,"
                f" Repeated or PipelineSpace, got {new_param!r}."
            )
        param_name = name if name else f"param_{len(self.get_attrs()) + 1}"

        class NewSpace(PipelineSpace):
            pass

        NewSpace.__name__ = self.__class__.__name__

        new_pipeline = NewSpace()
        for exist_name, value in self.get_attrs().items():
            setattr(new_pipeline, exist_name, value)
            if exist_name == param_name and not _parameters_are_equivalent(
                value, new_param
            ):
                raise ValueError(
                    f"A different parameter with the name {param_name!r} already exists"
                    " in the pipeline:\n"
                    f" {value}\n"
                    f" {new_param}"
                )
        if not hasattr(new_pipeline, param_name):
            setattr(new_pipeline, param_name, new_param)
        return new_pipeline

    def remove(self, name: str) -> PipelineSpace:
        """Remove a parameter from the pipeline by its name. This is NOT an in-place
            operation.

        Args:
            name: The name of the parameter to be removed.

        Returns:
            A NEW PipelineSpace without the removed parameter.

        Raises:
            ValueError: If no parameter with the specified name exists in the pipeline.
        """
        if name not in self.get_attrs():
            raise ValueError(
                f"No parameter with the name {name!r} exists in the pipeline."
            )

        class NewSpace(PipelineSpace):
            pass

        NewSpace.__name__ = self.__class__.__name__
        new_pipeline = NewSpace()
        for exist_name, value in self.get_attrs().items():
            if exist_name != name:
                setattr(new_pipeline, exist_name, value)

        return new_pipeline

    def add_prior(
        self,
        parameter_name: str,
        prior: Any,
        prior_confidence: ConfidenceLevel | Literal["low", "medium", "high"],
    ) -> PipelineSpace:
        """Add a prior to a parameter in the pipeline. This is NOT an in-place operation.

        Args:
            parameter_name: The name of the parameter to which the prior will be added.
            prior: The value of the prior to be added.
            prior_confidence: The confidence level of the prior, which can be "low",
                "medium", or "high".

        Returns:
            A NEW PipelineSpace with the added prior.

        Raises:
            ValueError: If no parameter with the specified name exists in the pipeline
                or if the parameter type does not support priors.
        """
        if parameter_name not in self.get_attrs():
            raise ValueError(
                f"No parameter with the name {parameter_name!r} exists in the pipeline."
            )

        class NewSpace(PipelineSpace):
            pass

        NewSpace.__name__ = self.__class__.__name__
        new_pipeline = NewSpace()
        for exist_name, value in self.get_attrs().items():
            if exist_name == parameter_name:
                if isinstance(value, Integer | Float | Categorical):
                    if value.has_prior:
                        raise ValueError(
                            f"The parameter {parameter_name!r} already has a prior:"
                            f" {value.prior!r}."
                        )
                    if isinstance(prior_confidence, str):
                        prior_confidence = convert_confidence_level(prior_confidence)
                    old_attributes = dict(value.get_attrs())
                    old_attributes["prior"] = prior
                    old_attributes["prior_confidence"] = prior_confidence
                    new_value = value.from_attrs(attrs=old_attributes)
                else:
                    raise ValueError(
                        f"The parameter {parameter_name!r} is of type"
                        f" {type(value).__name__}, which does not support priors."
                    )
            else:
                new_value = value
            setattr(new_pipeline, exist_name, new_value)
        return new_pipeline


class ConfidenceLevel(enum.Enum):
    """Enum representing confidence levels for sampling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def convert_confidence_level(confidence: str) -> ConfidenceLevel:
    """Convert a string representation of confidence level to ConfidenceLevel enum.

    Args:
        confidence: A string representing the confidence level, e.g., "low", "medium",
            "high".

    Returns:
        ConfidenceLevel: The corresponding ConfidenceLevel enum value.

    Raises:
        ValueError: If the input string does not match any of the defined confidence
            levels.
    """
    try:
        return ConfidenceLevel[confidence.upper()]
    except KeyError as e:
        raise ValueError(f"Invalid confidence level: {confidence}") from e


class Domain(Resolvable, abc.ABC, Generic[T]):
    """An abstract base class representing a domain in NePS spaces."""

    @property
    @abc.abstractmethod
    def min_value(self) -> T:
        """Get the minimum value of the domain."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def max_value(self) -> T:
        """Get the maximum value of the domain."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def has_prior(self) -> bool:
        """Check if the domain has a prior defined."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prior(self) -> T:
        """Get the prior value of the domain.
        Raises ValueError if the domain has no prior defined.

        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior.
        Raises ValueError if the domain has no prior defined.

        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the domain.
        This identifier is used to check if two domains are compatible based on their
        ranges.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> T:
        """Sample a value from the domain.
        Returns a value of type T that is within the domain's range.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def centered_around(
        self,
        center: T,
        confidence: ConfidenceLevel,
    ) -> Domain[T]:
        """Create a new domain centered around a given value with a specified confidence
        level.

        Args:
          center: The value around which to center the new domain.
          confidence: The confidence level for the new domain.
          center: T:
          confidence: ConfidenceLevel:

        Returns:
          A new Domain instance that is centered around the specified value.

        Raises:
          ValueError: If the center value is not within the domain's range.

        """
        raise NotImplementedError()

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the domain as a mapping.
        This method collects all attributes of the domain class and instance,
        excluding private attributes and methods, and returns them as a dictionary.

        Returns:
            A mapping of attribute names to their values.
        """
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def from_attrs(self, attrs: Mapping[str, Any]) -> Domain[T]:
        """Create a new Domain instance from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new Domain instance with the specified attributes.

        Raises:
            ValueError: If the attributes do not match the domain's expected structure.

        """
        return type(self)(**attrs)


def _calculate_new_domain_bounds(
    number_type: type[int] | type[float],
    min_value: int | float,
    max_value: int | float,
    center: int | float,
    confidence: ConfidenceLevel,
) -> tuple[int, int] | tuple[float, float]:
    """Calculate new bounds for a domain based on a center value and confidence level.
    This function determines the new minimum and maximum values for a domain based on
    a given center value and a confidence level. It splits the domain range into chunks
    and adjusts the bounds based on the specified confidence level.

    Args:
        number_type: The type of numbers in the domain (int or float).
        min_value: The minimum value of the domain.
        max_value: The maximum value of the domain.
        center: The center value around which to calculate the new bounds.
        confidence: The confidence level for the new bounds.

    Returns:
        A tuple containing the new minimum and maximum values for the domain.

    Raises:
        ValueError: If the center value is not within the domain's range or if the
        number_type is not supported.
    """
    if center < min_value or center > max_value:
        raise ValueError(
            f"Center value {center!r} must be within domain range [{min_value!r},"
            f" {max_value!r}]"
        )

    # Determine a chunk size by splitting the domain range into a fixed number of chunks.
    # Then use the confidence level to decide how many chunks to include
    # around the given center (on each side).

    number_of_chunks = 10.0
    chunk_size = (max_value - min_value) / number_of_chunks

    # The numbers refer to how many segments to have on each side of the center.
    # TODO: [lum] we need to make sure that in the end the range does not just have the
    # center, but at least a little bit more around it too.
    confidence_to_number_of_chunks_on_each_side = {
        ConfidenceLevel.HIGH: 1.0,
        ConfidenceLevel.MEDIUM: 2.5,
        ConfidenceLevel.LOW: 4.0,
    }

    chunk_multiplier = confidence_to_number_of_chunks_on_each_side[confidence]
    interval_radius = chunk_size * chunk_multiplier

    if number_type is int:
        # In this case we need to use ceil/floor so that we end up with ints.
        new_min = max(min_value, math.floor(center - interval_radius))
        new_max = min(max_value, math.ceil(center + interval_radius))
    elif number_type is float:
        new_min = max(min_value, center - interval_radius)
        new_max = min(max_value, center + interval_radius)
    else:
        raise ValueError(f"Unsupported number type {number_type!r}.")

    return new_min, new_max


class Categorical(Domain[int], Generic[T]):
    """A domain representing a categorical choice from a set of options.

    Attributes:
        choices: A tuple of choices or a Domain of choices.
        prior: The index of the prior choice in the choices tuple.
        prior_confidence: The confidence level of the prior choice.
    """

    def __init__(
        self,
        choices: tuple[T | Domain[T] | Resolvable | Any, ...] | Domain[T] | Resolvable,
        prior: int | Domain[int] | _Unset = _UNSET,
        prior_confidence: (
            ConfidenceLevel | Literal["low", "medium", "high"] | _Unset
        ) = _UNSET,
    ):
        """Initialize the Categorical domain with choices and optional prior.

        Args:
            choices: A tuple of choices or a Domain of choices.
            prior: The index of the prior choice in the choices tuple.
            prior_confidence: The confidence level of the prior choice.

        """
        self._choices: tuple[T | Domain[T] | Resolvable | Any, ...] | Domain[T]
        if isinstance(choices, Sequence):
            self._choices = tuple(choice for choice in choices)
            if any(isinstance(choice, tuple) for choice in self._choices) and any(
                not isinstance(choice, tuple) for choice in self._choices
            ):
                self._choices = tuple(
                    (choice,) if not isinstance(choice, tuple) else choice
                    for choice in self._choices
                )
        else:
            self._choices = choices  # type: ignore[assignment]
        self._prior = prior
        self._prior_confidence = (
            convert_confidence_level(prior_confidence)
            if isinstance(prior_confidence, str)
            else prior_confidence
        )
        if self._prior is not _UNSET and self._prior_confidence is _UNSET:
            raise ValueError(
                "If prior is set, prior_confidence must also be set to a valid value."
            )

    def __str__(self) -> str:
        """Get a string representation of the categorical domain."""
        string = f"Categorical(choices={self._choices!s}"
        if self.has_prior:
            string += f", prior={self._prior}, prior_confidence={self._prior_confidence}"
        string += ")"
        return string

    def compare_domain_to(self, other: object) -> bool:
        """Check if this categorical parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Categorical):
            return False
        return (
            self._prior == other._prior
            and self._prior_confidence == other._prior_confidence
            and self.choices == other.choices
        )

    @property
    def min_value(self) -> int:
        """Get the minimum value of the categorical domain.

        Returns:
            The minimum index of the choices, which is always 0.

        """
        return 0

    @property
    def max_value(self) -> int:
        """Get the maximum value of the categorical domain.

        Returns:
            The maximum index of the choices, which is the length of the choices tuple
            minus one.

        """
        return max(len(cast(tuple, self._choices)) - 1, 0)

    @property
    def choices(self) -> tuple[T | Domain[T] | Resolvable, ...] | Domain[T]:
        """Get the choices available in the categorical domain.

        Returns:
            A tuple of choices or a Domain of choices.

        """
        return self._choices

    @property
    def has_prior(self) -> bool:
        """Check if the categorical domain has a prior defined.

        Returns:
            True if the prior and prior confidence are set, False otherwise.
        """
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        """Get the prior index of the categorical domain.

        Returns:
          The index of the prior choice in the choices tuple.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence defined.")
        return int(cast(int, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior choice.

        Returns:
          The confidence level of the prior choice.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the categorical domain.

        Returns:
            A string representation of the number of choices in the domain.

        """
        return f"{len(cast(tuple, self._choices))}"

    def sample(self) -> int:
        """Sample a random index from the categorical choices.

        Returns:
          A randomly selected index from the choices tuple.

        Raises:
          ValueError: If the choices are empty.

        """
        return int(random.randint(0, len(cast(tuple[T], self._choices)) - 1))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Categorical:
        """Create a new categorical domain centered around a specific choice index.

        Args:
          center: The index of the choice around which to center the new domain.
          confidence: The confidence level for the new domain.
          center: int:
          confidence: ConfidenceLevel:

        Returns:
          A new Categorical instance with a range centered around the specified
          choice index.

        Raises:
          ValueError: If the center index is out of bounds of the choices.

        """
        new_min, new_max = cast(
            tuple[int, int],
            _calculate_new_domain_bounds(
                number_type=int,
                min_value=self.min_value,
                max_value=self.max_value,
                center=center,
                confidence=confidence,
            ),
        )
        new_choices = cast(tuple, self._choices)[new_min : new_max + 1]
        return Categorical(
            choices=new_choices,
            prior=new_choices.index(cast(tuple, self._choices)[center]),
            prior_confidence=confidence,
        )


class Float(Domain[float]):
    """A domain representing a continuous range of floating-point values.

    Attributes:
        min_value: The minimum value of the domain.
        max_value: The maximum value of the domain.
        log: Whether to sample values on a logarithmic scale.
        prior: The prior value for the domain, if any.
        prior_confidence: The confidence level of the prior value.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        log: bool = False,  # noqa: FBT001, FBT002
        prior: float | _Unset = _UNSET,
        prior_confidence: (
            Literal["low", "medium", "high"] | ConfidenceLevel | _Unset
        ) = _UNSET,
    ):
        """Initialize the Float domain with min and max values, and optional prior.

        Args:
            min_value: The minimum value of the domain.
            max_value: The maximum value of the domain.
            log: Whether to sample values on a logarithmic scale.
            prior: The prior value for the domain, if any.
            prior_confidence: The confidence level of the prior value.

        """
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = (
            convert_confidence_level(prior_confidence)
            if isinstance(prior_confidence, str)
            else prior_confidence
        )
        if self._prior is not _UNSET and self._prior_confidence is _UNSET:
            raise ValueError(
                "If prior is set, prior_confidence must also be set to a valid value."
            )

    def __str__(self) -> str:
        """Get a string representation of the floating-point domain."""
        string = f"Float({self._min_value}, {self._max_value}"
        if self._log:
            string += ", log"
        if self.has_prior:
            string += f", prior={self._prior}, prior_confidence={self._prior_confidence}"
        string += ")"
        return string

    def compare_domain_to(self, other: object) -> bool:
        """Check if this float parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Float):
            return False
        return (
            self._prior == other._prior
            and self._prior_confidence == other._prior_confidence
            and self.min_value == other.min_value
            and self.max_value == other.max_value
            and self._log == other._log
        )

    @property
    def min_value(self) -> float:
        """Get the minimum value of the floating-point domain.

        Returns:
          The minimum value of the domain.

        Raises:
          ValueError: If min_value is greater than max_value.

        """
        return self._min_value

    @property
    def max_value(self) -> float:
        """Get the maximum value of the floating-point domain.

        Returns:
          The maximum value of the domain.

        Raises:
          ValueError: If min_value is greater than max_value.

        """
        return self._max_value

    @property
    def has_prior(self) -> bool:
        """Check if the floating-point domain has a prior defined.

        Returns:
            True if the prior and prior confidence are set, False otherwise.

        """
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> float:
        """Get the prior value of the floating-point domain.

        Returns:
          The prior value of the domain.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence defined.")
        return float(cast(float, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior value.

        Returns:
          The confidence level of the prior value.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence  defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the floating-point
        domain.

        Returns:
            A string representation of the minimum and maximum values, and whether
            the domain is logarithmic.

        """
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> float:
        """Sample a random floating-point value from the domain.

        Returns:
          A randomly selected floating-point value within the domain's range.

        Raises:
          ValueError: If min_value is greater than max_value.

        """
        if self._log:
            log_min = math.log(self._min_value)
            log_max = math.log(self._max_value)
            return float(math.exp(random.uniform(log_min, log_max)))
        return float(random.uniform(self._min_value, self._max_value))

    def centered_around(
        self,
        center: float,
        confidence: ConfidenceLevel,
    ) -> Float:
        """Create a new floating-point domain centered around a specific value.

        Args:
            center: The value around which to center the new domain.
            confidence: The confidence level for the new domain.
            center: float:
            confidence: ConfidenceLevel:

        Returns:
            A new Float instance that is centered around the specified value.

        Raises:
            ValueError: If the center value is not within the domain's range.

        """
        new_min, new_max = _calculate_new_domain_bounds(
            number_type=float,
            min_value=self.min_value,
            max_value=self.max_value,
            center=center,
            confidence=confidence,
        )
        return Float(
            min_value=new_min,
            max_value=new_max,
            log=self._log,
            prior=center,
            prior_confidence=confidence,
        )


class Integer(Domain[int]):
    """A domain representing a range of integer values.

    Attributes:
        min_value: The minimum value of the domain.
        max_value: The maximum value of the domain.
        log: Whether to sample values on a logarithmic scale.
        prior: The prior value for the domain, if any.
        prior_confidence: The confidence level of the prior value.
    """

    def __init__(
        self,
        min_value: int,
        max_value: int,
        log: bool = False,  # noqa: FBT001, FBT002
        prior: float | int | _Unset = _UNSET,
        prior_confidence: (
            Literal["low", "medium", "high"] | ConfidenceLevel | _Unset
        ) = _UNSET,
    ):
        """Initialize the Integer domain with min and max values, and optional prior.

        Args:
            min_value: The minimum value of the domain.
            max_value: The maximum value of the domain.
            log: Whether to sample values on a logarithmic scale.
            prior: The prior value for the domain, if any.
            prior_confidence: The confidence level of the prior value.
        """
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = (
            convert_confidence_level(prior_confidence)
            if isinstance(prior_confidence, str)
            else prior_confidence
        )
        if self._prior != _UNSET and self._prior_confidence is _UNSET:
            raise ValueError(
                "If prior is set, prior_confidence must also be set to a valid value."
            )

    def __str__(self) -> str:
        """Get a string representation of the integer domain."""
        string = f"Integer({self._min_value}, {self._max_value}"
        if self._log:
            string += ", log"
        if self.has_prior:
            string += f", prior={self._prior}, prior_confidence={self._prior_confidence}"
        string += ")"
        return string

    def compare_domain_to(self, other: object) -> bool:
        """Check if this integer parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Integer):
            return False
        return (
            self._prior == other._prior
            and self._prior_confidence == other._prior_confidence
            and self.min_value == other.min_value
            and self.max_value == other.max_value
            and self._log == other._log
        )

    @property
    def min_value(self) -> int:
        """Get the minimum value of the integer domain.

        Returns:
          The minimum value of the domain.

        Raises:
          ValueError: If min_value is greater than max_value.

        """
        return self._min_value

    @property
    def max_value(self) -> int:
        """Get the maximum value of the integer domain.

        Returns:
          The maximum value of the domain.

        Raises:
          ValueError: If min_value is greater than max_value.

        """
        return self._max_value

    @property
    def has_prior(self) -> bool:
        """Check if the integer domain has a prior defined.

        Returns:
            True if the prior and prior confidence are set, False otherwise.

        """
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        """Get the prior value of the integer domain.

        Returns:
          The prior value of the domain.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence defined.")
        return int(cast(int, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior value.

        Returns:
          The confidence level of the prior value.

        Raises:
          ValueError: If the domain has no prior defined.

        """
        if not self.has_prior:
            raise ValueError("Domain has no prior and prior_confidence defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the integer domain.

        Returns:
            A string representation of the minimum and maximum values, and whether
            the domain is logarithmic.

        """
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> int:
        """Sample a random integer value from the domain.

        Returns:
            A randomly selected integer value within the domain's range.

        """
        if self._log:
            return int(
                math.exp(
                    random.uniform(math.log(self._min_value), math.log(self._max_value))
                )
            )
        return int(random.randint(self._min_value, self._max_value))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Integer:
        """Create a new integer domain centered around a specific value.

        Args:
            center: The value around which to center the new domain.
            confidence: The confidence level for the new domain.
            center: int:
            confidence: ConfidenceLevel:

        Returns:
            A new Integer instance that is centered around the specified value.

        Raises:
            ValueError: If the center value is not within the domain's range.

        """
        new_min, new_max = cast(
            tuple[int, int],
            _calculate_new_domain_bounds(
                number_type=int,
                min_value=self.min_value,
                max_value=self.max_value,
                center=center,
                confidence=confidence,
            ),
        )
        return Integer(
            min_value=new_min,
            max_value=new_max,
            log=self._log,
            prior=center,
            prior_confidence=confidence,
        )


class Operation(Resolvable):
    """A class representing an operation in a NePS space.

    Attributes:
        operator: The operator to be used in the operation, can be a callable or a string.
        args: A sequence of arguments to be passed to the operator.
        kwargs: A mapping of keyword arguments to be passed to the operator.
    """

    def __init__(
        self,
        operator: Callable | str,
        args: Sequence[Any] | Resolvable | None = None,
        kwargs: Mapping[str, Any] | Resolvable | None = None,
    ):
        """Initialize the Operation with an operator, arguments, and keyword arguments.

        Args:
            operator: The operator to be used in the operation, can be a callable or a
                string.
            args: A sequence of arguments to be passed to the operator.
            kwargs: A mapping of keyword arguments to be passed to the operator.

        """
        self._operator = operator

        self._args: tuple[Any, ...] | Resolvable
        if not isinstance(args, Resolvable):
            self._args = tuple(args) if args else ()
        else:
            self._args = args

        self._kwargs: Mapping[str, Any] | Resolvable
        if not isinstance(kwargs, Resolvable):
            self._kwargs = kwargs if kwargs else {}
        else:
            self._kwargs = kwargs

    def __str__(self) -> str:
        """Get a string representation of the operation."""
        return (
            f"Operation(operator={self._operator!s}, args={self._args!s},"
            f" kwargs={self._kwargs!s})"
        )

    def compare_domain_to(self, other: object) -> bool:
        """Check if this operation parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Operation):
            return False
        return (
            self.operator == other.operator
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    @property
    def operator(self) -> Callable | str:
        """Get the operator of the operation.

        Returns:
            The operator, which can be a callable or a string.

        Raises:
            ValueError: If the operator is not callable or a string.

        """
        return self._operator

    @property
    def args(self) -> tuple[Any, ...]:
        """Get the arguments of the operation.

        Returns:
            A tuple of arguments to be passed to the operator.

        Raises:
            ValueError: If the args are not resolved to a tuple.
        """
        if isinstance(self._args, Resolvable):
            raise ValueError(
                f"Operation args contain unresolved Resolvable: {self._args!r}. "
                "The operation needs to be resolved before accessing args as a tuple."
            )
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Get the keyword arguments of the operation.

        Returns:
            A mapping of keyword arguments to be passed to the operator.

        Raises:
            ValueError: If the kwargs are not resolved to a mapping.
        """
        if isinstance(self._kwargs, Resolvable):
            raise ValueError(
                f"Operation kwargs contain unresolved Resolvable: {self._kwargs!r}. "
                "The operation needs to be resolved before accessing kwargs as a mapping."
            )
        return self._kwargs

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the operation as a mapping.
        This method collects all attributes of the operation class and instance,
        excluding private attributes and methods, and returns them as a dictionary.

        Returns:
            A mapping of attribute names to their values.

        """
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def from_attrs(self, attrs: Mapping[str, Any]) -> Operation:
        """Create a new Operation instance from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new Operation instance with the specified attributes.

        Raises:
            ValueError: If the attributes do not match the operation's expected structure.

        """
        return type(self)(**attrs)


# TODO: [lum] For tuples, lists and dicts,
#  should we make the behavior similar to other resolvables,
#  in that they will be cached and then we also need to use Resampled for them?


class Resampled(Resolvable):
    """A class representing a resampling operation in a NePS space.

    Attributes:
        source: The source of the resampling, which can be a resolvable object or a
            string.
    """

    def __init__(self, source: Resolvable | str):
        """Initialize the Resampled object with a source.

        Args:
            source: The source of the resampling, can be a resolvable object or a string.
        """
        self._source = source

    def __str__(self) -> str:
        return f"Resampled({self._source!s})"

    @property
    def source(self) -> Resolvable | str:
        """Get the source of the resampling.

        Returns:
            The source of the resampling, which can be a resolvable object or a string

        """
        return self._source

    @property
    def is_resampling_by_name(self) -> bool:
        """Check if the resampling is by name.

        Returns:
            True if the source is a string, indicating a resampling by name,
            False if the source is a resolvable object.

        """
        return isinstance(self._source, str)

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the resampling source as a mapping.

        Returns:
          A mapping of attribute names to their values.

        Raises:
          ValueError: If the resampling is by name or the source is not resolvable.

        """
        if self.is_resampling_by_name:
            raise ValueError(
                f"This is a resampling by name, can't get attrs from it: {self.source!r}."
            )
        if not isinstance(self._source, Resolvable):
            raise ValueError(
                f"Source should be a resolvable object. Is: {self._source!r}."
            )
        return self._source.get_attrs()

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        """Create a new resolvable object from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new resolvable object created from the specified attributes.

        Raises:
            ValueError: If the resampling is by name or the source is not resolvable.

        """
        if self.is_resampling_by_name:
            raise ValueError(
                "This is a resampling by name, can't create object for it:"
                f" {self.source!r}."
            )
        if not isinstance(self._source, Resolvable):
            raise ValueError(
                f"Source should be a resolvable object. Is: {self._source!r}."
            )
        return self._source.from_attrs(attrs)

    def compare_domain_to(self, other: object) -> bool:
        """Check if this resampled parameter is equivalent to another.

        This method provides comparison logic without interfering with Python's
        object identity system (unlike __eq__). Use this for functional comparisons
        like checking if parameters have the same configuration.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Resampled):
            return False
        return self.source == other.source


class Repeated(Resolvable):
    """A class representing a sequence where a resolvable
    is repeated a variable number of times.

    Attributes:
        count: The count how many times the content should be repeated.
        content: The content which will be repeated.
    """

    def __init__(
        self,
        count: int | Domain[int] | Resolvable,
        content: Resolvable | Any,
    ):
        """Initialize the Repeated object with a count and content.

        Args:
            count: The count how many times the content should be repeated.
            content: The content which will be repeated.
        """
        if isinstance(count, int) and count < 0:
            raise ValueError(f"The received repeat count is negative. Received {count!r}")

        self._count = count
        self._content = content

    @property
    def count(self) -> int | Domain[int] | Resolvable:
        """Get the count how many times the content should be repeated.

        Returns:
            The count how many times the content will be repeated.
        """
        return self._count

    @property
    def content(self) -> Resolvable | Any:
        """Get the content which will be repeated.

        Returns:
            The content which will be repeated.
        """
        return self._content

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the resolvable as a mapping.

        Returns:
          A mapping of attribute names to their values.
        """
        return {"count": self.count, "content": self.content}

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        """Create a new resolvable object from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new resolvable object created from the specified attributes.
        """
        return Repeated(count=attrs["count"], content=attrs["content"])


class Lazy(Resolvable):
    """A class representing a lazy operation in a NePS space.

    The purpose is to have the resolution process
    stop at the moment it gets to this object,
    preventing the resolution of the object it wraps.

    Attributes:
        content: The content held, which can be a resolvable object or a
            tuple or a string.
    """

    def __init__(self, content: Resolvable | tuple[Any] | str):
        """Initialize the Lazy object with content.

        Args:
            content: The content being held, which can be a resolvable object
                or a tuple or a string.
        """
        self._content = content

    @property
    def content(self) -> Resolvable | tuple[Any] | str:
        """Get the content being held.

        Returns:
            The content of the lazy resolvable, which can be a resolvable object
            or a tuple or a string.
        """
        return self._content

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the lazy resolvable as a mapping.

        Raises:
          ValueError: Always, since this operation does not make sense here.
        """
        raise ValueError(
            f"This is a lazy resolvable. Can't get attrs from it: {self.content!r}."
        )

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:  # noqa: ARG002
        """Create a new resolvable object from the given attributes.

        Args:
            attrs: A mapping of attribute names to their values.

        Returns:
            A new resolvable object created from the specified attributes.

        Raises:
          ValueError: Always, since this operation does not make sense here.


        """
        raise ValueError(
            f"This is a lazy resolvable. Can't create object for it: {self.content!r}."
        )


# TODO: [lum] all the `get_attrs` and `from_attrs` MUST NOT raise.
#  They should return the best representation of themselves that they can.
#  This is because all resolvable objects can be nested content of other
#  resolvable objects that in general will interact with them
#  through these two methods.
#  When they raise, then the traversal will not be possible.
