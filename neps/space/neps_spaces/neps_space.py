"""This module defines various classes and protocols for representing and manipulating
search spaces in NePS (Neural Parameter Search). It includes definitions for domains,
pipelines, operations, and fidelity, as well as utilities for sampling and resolving
search spaces.
"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import enum
import functools
import math
import random
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from neps.optimizers import optimizer
from neps.space.neps_spaces import config_string
from neps.space.neps_spaces.sampling import OnlyPredefinedValuesSampler, RandomSampler

T = TypeVar("T")
P = TypeVar("P", bound="Pipeline")


# -------------------------------------------------


class _Unset:
    pass


_UNSET = _Unset()


# -------------------------------------------------


@runtime_checkable
class Resolvable(Protocol):
    """A protocol for objects that can be resolved into attributes."""

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the resolvable object as a mapping."""
        raise NotImplementedError()

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        """Create a new resolvable object from the given attributes."""
        raise NotImplementedError()


def resolvable_is_fully_resolved(resolvable: Resolvable) -> bool:
    """Check if a resolvable object is fully resolved.
    A resolvable object is considered fully resolved if all its attributes are either
    not instances of Resolvable or are themselves fully resolved.
    """
    attr_objects = resolvable.get_attrs().values()
    return all(
        not isinstance(obj, Resolvable) or resolvable_is_fully_resolved(obj)
        for obj in attr_objects
    )


@runtime_checkable
class DomainSampler(Protocol):
    """A protocol for domain samplers that can sample from a given domain."""

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Sample a value from the given domain.
        :param domain_obj: The domain object to sample from.
        :param current_path: The current path in the resolution context.
        :return: A sampled value of type T from the domain.
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()


# -------------------------------------------------


class Pipeline(Resolvable):
    """A class representing a pipeline in NePS spaces.
    It contains attributes that can be resolved into a configuration string,
    and it can be used to sample configurations based on defined domains.
    """

    @property
    def fidelity_attrs(self) -> Mapping[str, Fidelity]:
        """Get the fidelity attributes of the pipeline. Fidelity attributes are special
        attributes that represent the fidelity of the pipeline.
        :return: A mapping of fidelity attribute names to Fidelity objects.
        """
        return {k: v for k, v in self.get_attrs().items() if isinstance(v, Fidelity)}

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the pipeline as a mapping.
        This method collects all attributes of the pipeline class and instance,
        excluding private attributes and methods, and returns them as a dictionary.
        :return: A mapping of attribute names to their values.
        """
        attrs = {}

        for attr_name, attr_value in vars(self.__class__).items():
            if attr_name.startswith("_") or callable(attr_value):
                continue
            attrs[attr_name] = attr_value

        for attr_name, attr_value in vars(self).items():
            if attr_name.startswith("_") or callable(attr_value):
                continue
            attrs[attr_name] = attr_value

        properties_to_ignore = ("fidelity_attrs",)
        for property_to_ignore in properties_to_ignore:
            attrs.pop(property_to_ignore, None)

        return attrs

    def from_attrs(self, attrs: Mapping[str, Any]) -> Pipeline:
        """Create a new Pipeline instance from the given attributes.
        :param attrs: A mapping of attribute names to their values.
        :return: A new Pipeline instance with the specified attributes.
        :raises ValueError: If the attributes do not match the pipeline's expected
        structure.
        """
        new_pipeline = Pipeline()
        for name, value in attrs.items():
            setattr(new_pipeline, name, value)
        return new_pipeline


class ConfidenceLevel(enum.Enum):
    """Enum representing confidence levels for sampling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Domain(Resolvable, abc.ABC, Generic[T]):
    """An abstract base class representing a domain in NePS spaces.
    It defines the properties and methods that all domains must implement,
    such as min and max values, sampling, and centered domains.
    """

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
        :param center: The value around which to center the new domain.
        :param confidence: The confidence level for the new domain.
        :return: A new Domain instance that is centered around the specified value.
        :raises ValueError: If the center value is not within the domain's range.
        """
        raise NotImplementedError()

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the domain as a mapping.
        This method collects all attributes of the domain class and instance,
        excluding private attributes and methods, and returns them as a dictionary.
        :return: A mapping of attribute names to their values.
        """
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def from_attrs(self, attrs: Mapping[str, Any]) -> Domain[T]:
        """Create a new Domain instance from the given attributes.
        :param attrs: A mapping of attribute names to their values.
        :return: A new Domain instance with the specified attributes.
        :raises ValueError: If the attributes do not match the domain's expected
        structure.
        """
        return type(self)(**attrs)


def _calculate_new_domain_bounds(
    number_type: type[int] | type[float],
    min_value: int | float,
    max_value: int | float,
    center: int | float,
    confidence: ConfidenceLevel,
) -> tuple[int, int] | tuple[float, float]:
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
    It allows for sampling from a predefined set of choices and can be centered around
    a specific choice with a given confidence level.
    :param choices: A tuple of choices or a Domain of choices.
    :param prior_index: The index of the prior choice in the choices tuple.
    :param prior_confidence: The confidence level of the prior choice.
    """

    def __init__(
        self,
        choices: tuple[T | Domain[T] | Resolvable | Any, ...] | Domain[T],
        prior_index: int | Domain[int] | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        """Initialize the Categorical domain with choices and optional prior.
        :param choices: A tuple of choices or a Domain of choices.
        :param prior_index: The index of the prior choice in the choices tuple.
        :param prior_confidence: The confidence level of the prior choice.
        :raises ValueError: If the choices are empty or prior_index is out of bounds.
        """
        self._choices: tuple[T | Domain[T] | Resolvable | Any, ...] | Domain[T]
        if isinstance(choices, Sequence):
            self._choices = tuple(choice for choice in choices)
        else:
            self._choices = choices
        self._prior_index = prior_index
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> int:
        """Get the minimum value of the categorical domain.
        :return: The minimum index of the choices, which is always 0.
        """
        return 0

    @property
    def max_value(self) -> int:
        """Get the maximum value of the categorical domain.
        :return: The maximum index of the choices, which is the length of choices minus 1.
        """
        return max(len(cast(tuple, self._choices)) - 1, 0)

    @property
    def choices(self) -> tuple[T | Domain[T] | Resolvable, ...] | Domain[T]:
        """Get the choices available in the categorical domain.
        :return: A tuple of choices or a Domain of choices.
        """
        return self._choices

    @property
    def has_prior(self) -> bool:
        """Check if the categorical domain has a prior defined.
        :return: True if the prior index and confidence are set, False otherwise.
        """
        return self._prior_index is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        """Get the prior index of the categorical domain.
        :return: The index of the prior choice in the choices tuple.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return int(cast(int, self._prior_index))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior choice.
        :return: The confidence level of the prior choice.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the categorical domain.
        :return: A string representation of the number of choices in the domain.
        """
        return f"{len(cast(tuple, self._choices))}"

    def sample(self) -> int:
        """Sample a random index from the categorical choices.
        :return: A randomly selected index from the choices tuple.
        :raises ValueError: If the choices are empty.
        """
        return int(random.randint(0, len(cast(tuple[T], self._choices)) - 1))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Categorical:
        """Create a new categorical domain centered around a specific choice index.
        :param center: The index of the choice around which to center the new domain.
        :param confidence: The confidence level for the new domain.
        :return: A new Categorical instance with a range centered around the specified
        choice index.
        :raises ValueError: If the center index is out of bounds of the choices.
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
            prior_index=new_choices.index(cast(tuple, self._choices)[center]),
            prior_confidence=confidence,
        )


class Float(Domain[float]):
    """A domain representing a continuous range of floating-point values.
    It allows for sampling from a range defined by minimum and maximum values,
    and can be centered around a specific value with a given confidence level.
    :param min_value: The minimum value of the domain.
    :param max_value: The maximum value of the domain.
    :param log: Whether to sample values on a logarithmic scale.
    :param prior: The prior value for the domain, if any.
    :param prior_confidence: The confidence level of the prior value.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        log: bool = False,  # noqa: FBT001, FBT002
        prior: float | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        """Initialize the Float domain with min and max values, and optional prior.
        :param min_value: The minimum value of the domain.
        :param max_value: The maximum value of the domain.
        :param log: Whether to sample values on a logarithmic scale.
        :param prior: The prior value for the domain, if any.
        :param prior_confidence: The confidence level of the prior value.
        :raises ValueError: If min_value is greater than max_value.
        """
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> float:
        """Get the minimum value of the floating-point domain.
        :return: The minimum value of the domain.
        :raises ValueError: If min_value is greater than max_value.
        """
        return self._min_value

    @property
    def max_value(self) -> float:
        """Get the maximum value of the floating-point domain.
        :return: The maximum value of the domain.
        :raises ValueError: If min_value is greater than max_value.
        """
        return self._max_value

    @property
    def has_prior(self) -> bool:
        """Check if the floating-point domain has a prior defined.
        :return: True if the prior and prior confidence are set, False otherwise.
        """
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> float:
        """Get the prior value of the floating-point domain.
        :return: The prior value of the domain.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return float(cast(float, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior value.
        :return: The confidence level of the prior value.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the floating-point
        domain.
        :return: A string representation of the minimum and maximum values, and whether
        the domain is logarithmic.
        """
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> float:
        """Sample a random floating-point value from the domain.
        :return: A randomly selected floating-point value within the domain's range.
        :raises ValueError: If min_value is greater than max_value.
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
        :param center: The value around which to center the new domain.
        :param confidence: The confidence level for the new domain.
        :return: A new Float instance that is centered around the specified value.
        :raises ValueError: If the center value is not within the domain's range.
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
    It allows for sampling from a range defined by minimum and maximum values,
    and can be centered around a specific value with a given confidence level.
    :param min_value: The minimum value of the domain.
    :param max_value: The maximum value of the domain.
    :param log: Whether to sample values on a logarithmic scale.
    :param prior: The prior value for the domain, if any.
    :param prior_confidence: The confidence level of the prior value.
    """

    def __init__(
        self,
        min_value: int,
        max_value: int,
        log: bool = False,  # noqa: FBT001, FBT002
        prior: int | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        """Initialize the Integer domain with min and max values, and optional prior.
        :param min_value: The minimum value of the domain.
        :param max_value: The maximum value of the domain.
        :param log: Whether to sample values on a logarithmic scale.
        :param prior: The prior value for the domain, if any.
        :param prior_confidence: The confidence level of the prior value.
        :raises ValueError: If min_value is greater than max_value.
        """
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> int:
        """Get the minimum value of the integer domain.
        :return: The minimum value of the domain.
        :raises ValueError: If min_value is greater than max_value.
        """
        return self._min_value

    @property
    def max_value(self) -> int:
        """Get the maximum value of the integer domain.
        :return: The maximum value of the domain.
        :raises ValueError: If min_value is greater than max_value.
        """
        return self._max_value

    @property
    def has_prior(self) -> bool:
        """Check if the integer domain has a prior defined.
        :return: True if the prior and prior confidence are set, False otherwise.
        """
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        """Get the prior value of the integer domain.
        :return: The prior value of the domain.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return int(cast(int, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        """Get the confidence level of the prior value.
        :return: The confidence level of the prior value.
        :raises ValueError: If the domain has no prior defined.
        """
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        """Get a string identifier for the range compatibility of the integer domain.
        :return: A string representation of the minimum and maximum values, and whether
        the domain is logarithmic.
        """
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> int:
        """Sample a random integer value from the domain.
        :return: A randomly selected integer value within the domain's range.
        :raises NotImplementedError: If the domain is set to sample on a logarithmic
        scale, as this is not implemented yet.
        """
        if self._log:
            raise NotImplementedError("TODO.")
        return int(random.randint(self._min_value, self._max_value))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Integer:
        """Create a new integer domain centered around a specific value.
        :param center: The value around which to center the new domain.
        :param confidence: The confidence level for the new domain.
        :return: A new Integer instance that is centered around the specified value.
        :raises ValueError: If the center value is not within the domain's range.
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
    It encapsulates an operator (a callable or a string), arguments, and keyword
    arguments.
    The operator can be a function or a string representing a function name.
    :param operator: The operator to be used in the operation, can be a callable or a
    string.
    :param args: A sequence of arguments to be passed to the operator.
    :param kwargs: A mapping of keyword arguments to be passed to the operator.
    """

    def __init__(
        self,
        operator: Callable | str,
        args: Sequence[Any] | Resolvable | None = None,
        kwargs: Mapping[str, Any] | Resolvable | None = None,
    ):
        """Initialize the Operation with an operator, arguments, and keyword arguments.
        :param operator: The operator to be used in the operation, can be a callable or a
        string.
        :param args: A sequence of arguments to be passed to the operator.
        :param kwargs: A mapping of keyword arguments to be passed to the operator.
        :raises ValueError: If the operator is not callable or a string.
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

    @property
    def operator(self) -> Callable | str:
        """Get the operator of the operation.
        :return: The operator, which can be a callable or a string.
        :raises ValueError: If the operator is not callable or a string.
        """
        return self._operator

    @property
    def args(self) -> tuple[Any, ...]:
        """Get the arguments of the operation.
        :return: A tuple of arguments to be passed to the operator.
        :raises ValueError: If the args are not a tuple or Resolvable.
        """
        return cast(tuple[Any, ...], self._args)

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Get the keyword arguments of the operation.
        :return: A mapping of keyword arguments to be passed to the operator.
        :raises ValueError: If the kwargs are not a mapping or Resolvable.
        """
        return cast(Mapping[str, Any], self._kwargs)

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the operation as a mapping.
        This method collects all attributes of the operation class and instance,
        excluding private attributes and methods, and returns them as a dictionary.
        :return: A mapping of attribute names to their values.
        """
        # TODO: [lum] simplify this. We know the fields. Maybe other places too.
        result: dict[str, Any] = {}
        for name, value in vars(self).items():
            stripped_name = name.lstrip("_")
            if isinstance(value, dict):
                for k, v in value.items():
                    # Multiple {{}} needed to escape surrounding '{' and '}'.
                    result[f"{stripped_name}{{{k}}}"] = v
            elif isinstance(value, tuple):
                for i, v in enumerate(value):
                    result[f"{stripped_name}[{i}]"] = v
            else:
                result[stripped_name] = value
        return result

    def from_attrs(self, attrs: Mapping[str, Any]) -> Operation:
        """Create a new Operation instance from the given attributes.
        :param attrs: A mapping of attribute names to their values.
        :return: A new Operation instance with the specified attributes.
        :raises ValueError: If the attributes do not match the operation's expected
        structure.
        """
        # TODO: [lum] simplify this. We know the fields. Maybe other places too.
        final_attrs: dict[str, Any] = {}
        for name, value in attrs.items():
            if "{" in name and "}" in name:
                base, key = name.split("{")
                key = key.rstrip("}")
                final_attrs.setdefault(base, {})[key] = value
            elif "[" in name and "]" in name:
                base, idx_str = name.split("[")
                idx = int(idx_str.rstrip("]"))
                final_attrs.setdefault(base, []).insert(idx, value)
            else:
                final_attrs[name] = value
        return type(self)(**final_attrs)


class Resampled(Resolvable):
    """A class representing a resampling operation in a NePS space.
    It can either be a resolvable object or a string representing a resampling by name.
    :param source: The source of the resampling, can be a resolvable object or a string.
    """

    def __init__(self, source: Resolvable | str):
        """Initialize the Resampled object with a source.
        :param source: The source of the resampling, which can be a resolvable object or
        a string.
        :raises ValueError: If the source is not a resolvable object or a string.
        """
        self._source = source

    @property
    def source(self) -> Resolvable | str:
        """Get the source of the resampling.
        :return: The source of the resampling, which can be a resolvable object or a
        string.
        """
        return self._source

    @property
    def is_resampling_by_name(self) -> bool:
        """Check if the resampling is by name.
        :return: True if the source is a string, False otherwise.
        """
        return isinstance(self._source, str)

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the resampling source as a mapping.
        :return: A mapping of attribute names to their values.
        :raises ValueError: If the resampling is by name or the source is not resolvable.
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
        :param attrs: A mapping of attribute names to their values.
        :return: A new resolvable object created from the specified attributes.
        :raises ValueError: If the resampling is by name or the source is not resolvable.
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


class Fidelity(Resolvable, Generic[T]):
    """A class representing a fidelity in a NePS space.
    It encapsulates a domain that defines the range of values for the fidelity.
    :param domain: The domain of the fidelity, which can be an Integer or Float domain.
    :raises ValueError: If the domain has a prior defined, as fidelity domains should not
    have priors.
    """

    def __init__(self, domain: Integer | Float):
        """Initialize the Fidelity with a domain.
        :param domain: The domain of the fidelity, which can be an Integer or Float
        domain.
        :raises ValueError: If the domain has a prior defined, as fidelity domains should
        not have priors.
        """
        if domain.has_prior:
            raise ValueError(f"The domain of a Fidelity can not have priors: {domain!r}.")
        self._domain = domain

    @property
    def min_value(self) -> int | float:
        """Get the minimum value of the fidelity domain.
        :return: The minimum value of the fidelity domain.
        """
        return self._domain.min_value

    @property
    def max_value(self) -> int | float:
        """Get the maximum value of the fidelity domain.
        :return: The maximum value of the fidelity domain.
        """
        return self._domain.max_value

    def get_attrs(self) -> Mapping[str, Any]:
        """Get the attributes of the fidelity as a mapping.
        This method collects all attributes of the fidelity class and instance,
        excluding private attributes and methods, and returns them as a dictionary.
        :return: A mapping of attribute names to their values.
        :raises ValueError: If the fidelity has no domain defined.
        """
        raise ValueError("For a Fidelity object there is nothing to resolve.")

    def from_attrs(self, attrs: Mapping[str, Any]) -> Fidelity:  # noqa: ARG002
        """Create a new Fidelity instance from the given attributes.
        :param attrs: A mapping of attribute names to their values.
        :return: A new Fidelity instance with the specified attributes.
        :raises ValueError: If the fidelity has no domain defined.
        """
        raise ValueError("For a Fidelity object there is nothing to resolve.")


# -------------------------------------------------


class SamplingResolutionContext:
    """A context for resolving samplings in a NePS space.
    It manages the resolution root, domain sampler, environment values,
    and keeps track of samplings made and resolved objects.
    :param resolution_root: The root of the resolution, which should be a Resolvable
    object.
    :param domain_sampler: The DomainSampler to use for sampling from Domain objects.
    :param environment_values: A mapping of environment values that are fixed and not
    related
    to samplings. These values can be used in the resolution process.
    :raises ValueError: If the resolution_root is not a Resolvable, or if the
    domain_sampler is not a DomainSampler, or if the environment_values is not a Mapping.
    """

    def __init__(
        self,
        *,
        resolution_root: Resolvable,
        domain_sampler: DomainSampler,
        environment_values: Mapping[str, Any],
    ):
        """Initialize the SamplingResolutionContext with a resolution root, domain
        sampler, and environment values.
        :param resolution_root: The root of the resolution, which should be a Resolvable
        object.
        :param domain_sampler: The DomainSampler to use for sampling from Domain objects.
        :param environment_values: A mapping of environment values that are fixed and not
        related to samplings. These values can be used in the resolution process.
        :raises ValueError: If the resolution_root is not a Resolvable, or if the
        domain_sampler is not a DomainSampler, or if the environment_values is not a
        Mapping.
        """
        if not isinstance(resolution_root, Resolvable):
            raise ValueError(
                "The received `resolution_root` is not a Resolvable:"
                f" {resolution_root!r}."
            )

        if not isinstance(domain_sampler, DomainSampler):
            raise ValueError(
                "The received `domain_sampler` is not a DomainSampler:"
                f" {domain_sampler!r}."
            )

        if not isinstance(environment_values, Mapping):
            raise ValueError(
                "The received `environment_values` is not a Mapping:"
                f" {environment_values!r}."
            )

        # `_resolution_root` stores the root of the resolution.
        self._resolution_root: Resolvable = resolution_root

        # `_domain_sampler` stores the object responsible for sampling from Domain
        # objects.
        self._domain_sampler = domain_sampler

        # # `_environment_values` stores fixed values from outside.
        # # They are not related to samplings and can not be mutated or similar.
        self._environment_values = environment_values

        # `_samplings_made` stores the values we have sampled
        # and can be used later in case we want to redo a resolving.
        self._samplings_made: dict[str, Any] = {}

        # `_resolved_objects` stores the intermediate values to make re-use possible.
        self._resolved_objects: dict[Any, Any] = {}

        # `_current_path_parts` stores the current path we are resolving.
        self._current_path_parts: list[str] = []

    @property
    def resolution_root(self) -> Resolvable:
        """Get the root of the resolution.
        :return: The root of the resolution, which should be a Resolvable object.
        """
        return self._resolution_root

    @property
    def samplings_made(self) -> Mapping[str, Any]:
        """Get the samplings made during the resolution process.
        :return: A mapping of paths to sampled values.
        """
        return self._samplings_made

    @property
    def environment_values(self) -> Mapping[str, Any]:
        """Get the environment values that are fixed and not related to samplings.
        :return: A mapping of environment variable names to their values.
        """
        return self._environment_values

    @contextlib.contextmanager
    def resolving(self, _obj: Any, name: str) -> Generator[None]:
        """Context manager for resolving an object in the current resolution context.
        :param _obj: The object being resolved, can be any type.
        :param name: The name of the object being resolved, used for debugging.
        :raises ValueError: If the name is not a valid string.
        """
        if not name or not isinstance(name, str):
            raise ValueError(
                f"Given name for what we are resolving is invalid: {name!r}."
            )

        # It is possible that the received object has already been resolved.
        # That is expected and is okay, so no check is made for it.
        # For example, in the case of a Resampled we can receive the same object again.

        self._current_path_parts.append(name)
        try:
            yield
        finally:
            self._current_path_parts.pop()

    def was_already_resolved(self, obj: Any) -> bool:
        """Check if the given object was already resolved in the current context.
        :param obj: The object to check if it was already resolved.
        :return: True if the object was already resolved, False otherwise.
        """
        return obj in self._resolved_objects

    def add_resolved(self, original: Any, resolved: Any) -> None:
        """Add a resolved object to the context.
        :param original: The original object that was resolved.
        :param resolved: The resolved value of the original object.
        :raises ValueError: If the original object was already resolved or if it is a
        Resampled.
        """
        if self.was_already_resolved(original):
            raise ValueError(
                f"Original object has already been resolved: {original!r}. "
                + "\nIf you are doing resampling by name, "
                + "make sure you are not forgetting to request resampling also for"
                " related objects." + "\nOtherwise it could lead to infinite recursion."
            )
        if isinstance(original, Resampled):
            raise ValueError(
                f"Attempting to add a Resampled object to resolved values: {original!r}."
            )
        self._resolved_objects[original] = resolved

    def get_resolved(self, obj: Any) -> Any:
        """Get the resolved value for the given object.
        :param obj: The object for which to get the resolved value.
        :return: The resolved value of the object.
        :raises ValueError: If the object was not already resolved in the context.
        """
        try:
            return self._resolved_objects[obj]
        except KeyError as err:
            raise ValueError(
                f"Given object was not already resolved. Please check first: {obj!r}"
            ) from err

    def sample_from(self, domain_obj: Domain) -> Any:
        """Sample a value from the given domain object.
        :param domain_obj: The domain object from which to sample a value.
        :return: The sampled value from the domain object.
        :raises ValueError: If the domain object was already resolved or if the path
        has already been sampled from.
        """
        # Each `domain_obj` is only ever sampled from once.
        # This is okay and the expected behavior.
        # For each `domain_obj`, its sampled value is either directly stored itself,
        # or is used in some other Resolvable.
        # In both cases that sampled value is cached for later uses,
        # and so the `domain_obj` will not be re-sampled from again.
        if self.was_already_resolved(domain_obj):
            raise ValueError(
                "We have already sampled a value for the given domain object:"
                f" {domain_obj!r}." + "\nThis should not be happening."
            )

        # The range compatibility identifier is there to make sure when we say
        # the path matches, that the range for the value we are looking up also matches.
        domain_obj_type_name = type(domain_obj).__name__.lower()
        range_compatibility_identifier = domain_obj.range_compatibility_identifier
        domain_obj_identifier = (
            f"{domain_obj_type_name}__{range_compatibility_identifier}"
        )

        current_path = ".".join(self._current_path_parts)
        current_path += "::" + domain_obj_identifier

        if current_path in self._samplings_made:
            # We have already sampled a value for this path. This should not happen.
            # Every time we sample a domain, it should have its own different path.
            raise ValueError(
                f"We have already sampled a value for the current path: {current_path!r}."
                + "\nThis should not be happening."
            )

        sampled_value = self._domain_sampler(
            domain_obj=domain_obj,
            current_path=current_path,
        )

        self._samplings_made[current_path] = sampled_value
        return self._samplings_made[current_path]

    def get_value_from_environment(self, var_name: str) -> Any:
        """Get a value from the environment variables.
        :param var_name: The name of the environment variable to get the value from.
        :return: The value of the environment variable.
        :raises ValueError: If the environment variable is not found in the context.
        """
        try:
            return self._environment_values[var_name]
        except KeyError as err:
            raise ValueError(
                f"No value is available for the environment variable {var_name!r}."
            ) from err


class SamplingResolver:
    """A class responsible for resolving samplings in a NePS space.
    It uses a SamplingResolutionContext to manage the resolution process,
    and a DomainSampler to sample values from Domain objects.
    :param resolver: The resolver to use for resolving objects.
    This should be a callable that takes an object and a context and returns the resolved
    object.
    :raises ValueError: If the resolver is not a callable or if it is not a
    DomainSampler or a SamplingResolutionContext.
    """

    def __call__(
        self,
        obj: Resolvable,
        domain_sampler: DomainSampler,
        environment_values: Mapping[str, Any],
    ) -> tuple[Resolvable, SamplingResolutionContext]:
        """Resolve the given object in the context of the provided domain sampler and
        environment values.
        :param obj: The Resolvable object to resolve.
        :param domain_sampler: The DomainSampler to use for sampling from Domain objects.
        :param environment_values: A mapping of environment values that are fixed and not
        related to samplings.
        :return: A tuple containing the resolved object and the
        SamplingResolutionContext.
        :raises ValueError: If the object is not a Resolvable, or if the domain_sampler
        is not a DomainSampler, or if the environment_values is not a Mapping.
        """
        context = SamplingResolutionContext(
            resolution_root=obj,
            domain_sampler=domain_sampler,
            environment_values=environment_values,
        )
        return self._resolve(obj, "Resolvable", context), context

    def _resolve(self, obj: Any, name: str, context: SamplingResolutionContext) -> Any:
        with context.resolving(obj, name):
            return self._resolver_dispatch(obj, context)

    @functools.singledispatchmethod
    def _resolver_dispatch(
        self,
        any_obj: Any,
        _context: SamplingResolutionContext,
    ) -> Any:
        # Default resolver. To be used for types which are not instances of `Resolvable`.
        # No need to store or lookup from context, directly return the given object.
        if isinstance(any_obj, Resolvable):
            raise ValueError(
                "The default resolver is not supposed to be called for resolvable"
                f" objects. Received: {any_obj!r}."
            )
        return any_obj

    @_resolver_dispatch.register
    def _(
        self,
        pipeline_obj: Pipeline,
        context: SamplingResolutionContext,
    ) -> Any:
        if context.was_already_resolved(pipeline_obj):
            return context.get_resolved(pipeline_obj)

        initial_attrs = pipeline_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

        result = pipeline_obj
        if needed_resolving:
            result = pipeline_obj.from_attrs(final_attrs)

        context.add_resolved(pipeline_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        domain_obj: Domain,
        context: SamplingResolutionContext,
    ) -> Any:
        if context.was_already_resolved(domain_obj):
            return context.get_resolved(domain_obj)

        initial_attrs = domain_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

        resolved_domain_obj = domain_obj
        if needed_resolving:
            resolved_domain_obj = domain_obj.from_attrs(final_attrs)

        try:
            sampled_value = context.sample_from(resolved_domain_obj)
        except Exception as e:
            raise ValueError(f"Failed to sample from {resolved_domain_obj!r}.") from e
        result = self._resolve(sampled_value, "sampled_value", context)

        context.add_resolved(domain_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        categorical_obj: Categorical,
        context: SamplingResolutionContext,
    ) -> Any:
        if context.was_already_resolved(categorical_obj):
            return context.get_resolved(categorical_obj)

        # In the case of categorical choices, we may skip resolving each choice initially,
        # only after sampling we go into resolving whatever choice was chosen.
        # This avoids resolving things which won't be needed at all.
        # If the choices themselves come from some Resolvable, they will be resolved.

        initial_attrs = categorical_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            if attr_name == "choices":
                if isinstance(initial_attr_value, Resolvable):
                    # Resolving here like below works fine since the expectation
                    # is that we will get back a tuple of choices.
                    # Any element in that tuple can be a Resolvable,
                    # but will not be resolved from the call directly below,
                    # as the tuple is returned as is,
                    # without going into resolving its elements.
                    # If we add a `_resolve_tuple` functionality to go into tuples
                    # and resolve their contents, the call below will likely
                    # lead to too much work being done or issues.
                    resolved_attr_value = self._resolve(
                        initial_attr_value, attr_name, context
                    )
                else:
                    resolved_attr_value = initial_attr_value
            else:
                resolved_attr_value = self._resolve(
                    initial_attr_value, attr_name, context
                )
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

        resolved_categorical_obj = categorical_obj
        if needed_resolving:
            resolved_categorical_obj = cast(
                Categorical, categorical_obj.from_attrs(final_attrs)
            )

        try:
            sampled_index = context.sample_from(resolved_categorical_obj)
        except Exception as e:
            raise ValueError(
                f"Failed to sample from {resolved_categorical_obj!r}."
            ) from e
        sampled_value = cast(tuple, resolved_categorical_obj.choices)[sampled_index]
        result = self._resolve(sampled_value, "sampled_value", context)

        context.add_resolved(categorical_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        operation_obj: Operation,
        context: SamplingResolutionContext,
    ) -> Any:
        if context.was_already_resolved(operation_obj):
            return context.get_resolved(operation_obj)

        # It is possible that the `operation_obj` will require two runs to be fully
        # resolved. For example if `operation_obj.args` is not defined as a tuple of
        # args, but is a Resolvable that needs to be resolved first itself,
        # for us to have the actual tuple of args.

        # First run.

        initial_attrs = operation_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

        operation_obj_first_run = operation_obj
        if needed_resolving:
            operation_obj_first_run = operation_obj.from_attrs(final_attrs)

        # Second run.
        # It is possible the first run was enough,
        # in this case what we do below won't lead to any changes.

        initial_attrs = operation_obj_first_run.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

        operation_obj_second_run = operation_obj_first_run
        if needed_resolving:
            operation_obj_second_run = operation_obj_first_run.from_attrs(final_attrs)

        result = operation_obj_second_run

        context.add_resolved(operation_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        resampled_obj: Resampled,
        context: SamplingResolutionContext,
    ) -> Any:
        # The results of Resampled are never stored or looked up from cache
        # since it would break the logic of their expected behavior.
        # Particularly, when Resampled objects are nested (at any depth) inside of
        # other Resampled objects, adding them to the resolution context would result
        # in the resolution not doing the right thing.

        if resampled_obj.is_resampling_by_name:
            # We are dealing with a resampling by name,
            # We will first need to look up the source object referenced by name.
            # That will then be the object to resample.
            referenced_obj_name = cast(str, resampled_obj.source)
            referenced_obj = getattr(context.resolution_root, referenced_obj_name)
            resampled_obj = Resampled(referenced_obj)

        initial_attrs = resampled_obj.get_attrs()
        resolvable_to_resample_obj = resampled_obj.from_attrs(initial_attrs)

        type_name = type(resolvable_to_resample_obj).__name__.lower()
        return self._resolve(
            resolvable_to_resample_obj, f"resampled_{type_name}", context
        )

    @_resolver_dispatch.register
    def _(
        self,
        fidelity_obj: Fidelity,
        context: SamplingResolutionContext,
    ) -> Any:
        # A Fidelity object should only really be used in one place,
        # so we check if we have seen it before.
        # For that we will be storing its result in the resolved cache.
        if context.was_already_resolved(fidelity_obj):
            raise ValueError("Fidelity object reused multiple times in the pipeline.")

        # The way resolution works for Fidelity objects is that
        # we use the domain inside it only to know the bounds for valid values.
        # The actual value for the fidelity comes from the outside in the form of an
        # environment value, which we look up by the attribute name of the
        # received fidelity object inside the resolution root.

        names_for_this_fidelity_obj = [
            attr_name
            for attr_name, attr_value in context.resolution_root.get_attrs().items()
            if attr_value is fidelity_obj
        ]

        if len(names_for_this_fidelity_obj) == 0:
            raise ValueError(
                "A fidelity object should be a direct attribute of the pipeline."
            )
        if len(names_for_this_fidelity_obj) > 1:
            raise ValueError(
                "A fidelity object should only be referenced once in the pipeline."
            )

        fidelity_name = names_for_this_fidelity_obj[0]

        try:
            result = context.get_value_from_environment(fidelity_name)
        except ValueError as err:
            raise ValueError(
                "No value is available in the environment for fidelity"
                f" {fidelity_name!r}."
            ) from err

        if not fidelity_obj.min_value <= result <= fidelity_obj.max_value:
            raise ValueError(
                f"Value for fidelity with name {fidelity_name!r} is outside its allowed"
                " range "
                + f"[{fidelity_obj.min_value!r}, {fidelity_obj.max_value!r}]. "
                + f"Received: {result!r}."
            )

        context.add_resolved(fidelity_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: Resolvable,
        context: SamplingResolutionContext,  # noqa: ARG002
    ) -> Any:
        # Called when no specialized resolver was available for the specific resolvable
        # type. That is not something that is normally expected.
        raise ValueError(
            "No specialized resolver was registered for object of type"
            f" {type(resolvable_obj)!r}."
        )


def resolve(
    pipeline: P,
    domain_sampler: DomainSampler | None = None,
    environment_values: Mapping[str, Any] | None = None,
) -> tuple[P, SamplingResolutionContext]:
    """Resolve a NePS pipeline with the given domain sampler and environment values.
    :param pipeline: The pipeline to resolve, which should be a Pipeline object.
    :param domain_sampler: The DomainSampler to use for sampling from Domain objects.
    If None, a RandomSampler with no predefined values will be used.
    :param environment_values: A mapping of environment variable names to their values.
    If None, an empty mapping will be used.
    :return: A tuple containing the resolved pipeline and the SamplingResolutionContext.
    :raises ValueError: If the pipeline is not a Pipeline object or if the domain_sampler
    is not a DomainSampler or if the environment_values is not a Mapping.
    """
    if domain_sampler is None:
        # By default, use a random sampler with no predefined values.
        domain_sampler = RandomSampler(predefined_samplings={})

    if environment_values is None:
        # By default, have no environment values.
        environment_values = {}

    sampling_resolver = SamplingResolver()
    resolved_pipeline, context = sampling_resolver(
        obj=pipeline,
        domain_sampler=domain_sampler,
        environment_values=environment_values,
    )
    return cast(P, resolved_pipeline), context


# -------------------------------------------------


def convert_operation_to_callable(operation: Operation) -> Callable:
    """Convert an Operation to a callable that can be executed.
    :param operation: The Operation to convert.
    :return: A callable that represents the operation.
    :raises ValueError: If the operation is not a valid Operation object.
    """
    operator = cast(Callable, operation.operator)

    operation_args = []
    for arg in operation.args:
        operation_args.append(
            convert_operation_to_callable(arg) if isinstance(arg, Operation) else arg
        )

    operation_kwargs = {}
    for kwarg_name, kwarg_value in operation.kwargs.items():
        operation_kwargs[kwarg_name] = (
            convert_operation_to_callable(kwarg_value)
            if isinstance(kwarg_value, Operation)
            else kwarg_value
        )

    return cast(Callable, operator(*operation_args, **operation_kwargs))


def _operation_to_unwrapped_config(
    operation: Operation | str,
    level: int = 1,
) -> list[config_string.UnwrappedConfigStringPart]:
    result = []

    if isinstance(operation, Operation):
        operator = operation.operator
        kwargs = str(operation.kwargs)
        item = config_string.UnwrappedConfigStringPart(
            level=level,
            opening_index=-1,
            operator=operator,
            hyperparameters=kwargs,
            operands="",
        )
        result.append(item)
        for operand in operation.args:
            result.extend(_operation_to_unwrapped_config(operand, level + 1))
    else:
        item = config_string.UnwrappedConfigStringPart(
            level=level,
            opening_index=-1,
            operator=operation,
            hyperparameters="",
            operands="",
        )
        result.append(item)
    return result


def convert_operation_to_string(operation: Operation) -> str:
    """Convert an Operation to a string representation.
    :param operation: The Operation to convert.
    :return: A string representation of the operation.
    :raises ValueError: If the operation is not a valid Operation object.
    """
    unwrapped_config = tuple(_operation_to_unwrapped_config(operation))
    return config_string.wrap_config_into_string(unwrapped_config)


# -------------------------------------------------


class NepsCompatConverter:
    """A class to convert between NePS configurations and NEPS-compatible configurations.
    It provides methods to convert a SamplingResolutionContext to a NEPS-compatible config
    and to convert a NEPS-compatible config back to a SamplingResolutionContext.
    :param resolution_context: The SamplingResolutionContext to convert.
    :raises ValueError: If the resolution_context is not a SamplingResolutionContext.
    """

    _SAMPLING_PREFIX = "SAMPLING__"
    _ENVIRONMENT_PREFIX = "ENVIRONMENT__"
    _SAMPLING_PREFIX_LEN = len(_SAMPLING_PREFIX)
    _ENVIRONMENT_PREFIX_LEN = len(_ENVIRONMENT_PREFIX)

    @dataclasses.dataclass(frozen=True)
    class _FromNepsConfigResult:
        predefined_samplings: Mapping[str, Any]
        environment_values: Mapping[str, Any]
        extra_kwargs: Mapping[str, Any]

    @classmethod
    def to_neps_config(
        cls,
        resolution_context: SamplingResolutionContext,
    ) -> Mapping[str, Any]:
        """Convert a SamplingResolutionContext to a NEPS-compatible config.
        :param resolution_context: The SamplingResolutionContext to convert.
        :return: A mapping of NEPS-compatible configuration keys to their values.
        :raises ValueError: If the resolution_context is not a SamplingResolutionContext.
        """
        config: dict[str, Any] = {}

        samplings_made = resolution_context.samplings_made
        for sampling_path, value in samplings_made.items():
            config[f"{cls._SAMPLING_PREFIX}{sampling_path}"] = value

        environment_values = resolution_context.environment_values
        for env_name, value in environment_values.items():
            config[f"{cls._ENVIRONMENT_PREFIX}{env_name}"] = value

        return config

    @classmethod
    def from_neps_config(
        cls,
        config: Mapping[str, Any],
    ) -> _FromNepsConfigResult:
        """Convert a NEPS-compatible config to a SamplingResolutionContext.
        :param config: A mapping of NEPS-compatible configuration keys to their values.
        :return: A _FromNepsConfigResult containing predefined samplings,
        environment values, and extra kwargs.
        :raises ValueError: If the config is not a valid NEPS-compatible config.
        """
        predefined_samplings = {}
        environment_values = {}
        extra_kwargs = {}

        for name, value in config.items():
            if name.startswith(cls._SAMPLING_PREFIX):
                sampling_path = name[cls._SAMPLING_PREFIX_LEN :]
                predefined_samplings[sampling_path] = value
            elif name.startswith(cls._ENVIRONMENT_PREFIX):
                env_name = name[cls._ENVIRONMENT_PREFIX_LEN :]
                environment_values[env_name] = value
            else:
                extra_kwargs[name] = value

        return cls._FromNepsConfigResult(
            predefined_samplings=predefined_samplings,
            environment_values=environment_values,
            extra_kwargs=extra_kwargs,
        )


def _prepare_sampled_configs(
    chosen_pipelines: list[tuple[Pipeline, SamplingResolutionContext]],
    n_prev_trials: int,
    return_single: bool,  # noqa: FBT001
) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
    configs = []
    for i, (_resolved_pipeline, resolution_context) in enumerate(chosen_pipelines):
        neps_config = NepsCompatConverter.to_neps_config(
            resolution_context=resolution_context,
        )

        config = optimizer.SampledConfig(
            config=neps_config,
            id=str(n_prev_trials + i + 1),
            previous_config_id=None,
        )
        configs.append(config)

    if return_single:
        return configs[0]

    return configs


def adjust_evaluation_pipeline_for_neps_space(
    evaluation_pipeline: Callable,
    pipeline_space: P,
    operation_converter: Callable[[Operation], Any] = convert_operation_to_callable,
) -> Callable | str:
    """Adjust the evaluation pipeline to work with a NePS space.
    This function wraps the evaluation pipeline to sample from the NePS space
    and convert the sampled pipeline to a format compatible with the evaluation pipeline.
    :param evaluation_pipeline: The evaluation pipeline to adjust.
    :param pipeline_space: The NePS pipeline space to sample from.
    :param operation_converter: A callable to convert Operation objects to a format
    compatible with the evaluation pipeline.
    :return: A wrapped evaluation pipeline that samples from the NePS space.
    :raises ValueError: If the evaluation_pipeline is not callable or if the
    pipeline_space is not a Pipeline object.
    """

    @functools.wraps(evaluation_pipeline)
    def inner(*args: Any, **kwargs: Any) -> Any:
        # `kwargs` can contain other things not related to
        # the samplings to make or to environment values.
        # That is not an issue. Those items will be passed through.

        sampled_pipeline_data = NepsCompatConverter.from_neps_config(config=kwargs)

        sampled_pipeline, _resolution_context = resolve(
            pipeline=pipeline_space,
            domain_sampler=OnlyPredefinedValuesSampler(
                predefined_samplings=sampled_pipeline_data.predefined_samplings,
            ),
            environment_values=sampled_pipeline_data.environment_values,
        )

        config = dict(**sampled_pipeline.get_attrs())

        for name, value in config.items():
            if isinstance(value, Operation):
                config[name] = operation_converter(value)

        # So that we still pass the kwargs not related to the config,
        # start with the extra kwargs we passed to the converter.
        new_kwargs = dict(**sampled_pipeline_data.extra_kwargs)
        # Then add all the kwargs from the config.
        new_kwargs.update(config)

        return evaluation_pipeline(*args, **new_kwargs)

    return inner
