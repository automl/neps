from __future__ import annotations

import abc
import dataclasses
import functools
import heapq
import random
import math
import enum
import contextlib
from typing import (
    TypeVar,
    Generic,
    Sequence,
    Any,
    Protocol,
    runtime_checkable,
    cast,
    Callable,
    Mapping,
    Generator,
    Type,
)

import neps.space.new_space.config_string as config_string
import neps.optimizers.optimizer as optimizer
import neps.state.trial as trial_state
import neps.state.optimizer as optimizer_state


T = TypeVar("T")
P = TypeVar("P", bound="Pipeline")


# -------------------------------------------------


class _Unset:
    pass


_UNSET = _Unset()


# -------------------------------------------------


@runtime_checkable
class Resolvable(Protocol):
    def get_attrs(self) -> Mapping[str, Any]:
        raise NotImplementedError()

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        raise NotImplementedError()


def resolvable_is_fully_resolved(resolvable: Resolvable) -> bool:
    attr_objects = resolvable.get_attrs().values()
    return all(not isinstance(obj, Resolvable) or resolvable_is_fully_resolved(obj) for obj in attr_objects)


@runtime_checkable
class DomainSampler(Protocol):
    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        raise NotImplementedError()


# -------------------------------------------------


class Pipeline(Resolvable):
    @property
    def fidelity_attrs(self) -> Mapping[str, Fidelity]:
        return {k: v for k, v in self.get_attrs().items() if isinstance(v, Fidelity)}

    def get_attrs(self) -> Mapping[str, Any]:
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
        new_pipeline = Pipeline()
        for name, value in attrs.items():
            setattr(new_pipeline, name, value)
        return new_pipeline


class ConfidenceLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Domain(Resolvable, abc.ABC, Generic[T]):
    @property
    @abc.abstractmethod
    def min_value(self) -> T:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def max_value(self) -> T:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def has_prior(self) -> bool:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prior(self) -> T:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prior_confidence(self) -> ConfidenceLevel:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def range_compatibility_identifier(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def centered_around(
        self,
        center: T,
        confidence: ConfidenceLevel,
    ) -> Domain[T]:
        raise NotImplementedError()

    def get_attrs(self) -> Mapping[str, Any]:
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def from_attrs(self, attrs: Mapping[str, Any]) -> Domain[T]:
        return type(self)(**attrs)


def _calculate_new_domain_bounds(
    number_type: Type[int] | Type[float],
    min_value: int | float,
    max_value: int | float,
    center: int | float,
    confidence: ConfidenceLevel,
) -> tuple[int, int] | tuple[float, float]:
    if center < min_value or center > max_value:
        raise ValueError(f"Center value {center!r} must be within domain range [{min_value!r}, {max_value!r}]")

    # Determine a chunk size by splitting the domain range into a fixed number of chunks.
    # Then use the confidence level to decide how many chunks to include
    # around the given center (on each side).

    number_of_chunks = 10.0
    chunk_size = (max_value - min_value) / number_of_chunks

    # The numbers refer to how many segments to have on each side of the center.
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
    def __init__(
        self,
        choices: tuple[T | Domain[T] | Resolvable, ...] | Domain[T],
        prior_index: int | Domain[int] | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        self._choices: tuple[T | Domain[T] | Resolvable, ...] | Domain[T]
        if isinstance(choices, Sequence):
            self._choices = tuple(choice for choice in choices)
        else:
            self._choices = choices
        self._prior_index = prior_index
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> int:
        return 0

    @property
    def max_value(self) -> int:
        return max(len(cast(tuple, self._choices)) - 1, 0)

    @property
    def choices(self) -> tuple[T | Domain[T] | Resolvable, ...] | Domain[T]:
        return self._choices

    @property
    def has_prior(self) -> bool:
        return self._prior_index is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return int(cast(int, self._prior_index))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        return f"{len(cast(tuple, self._choices))}"

    def sample(self) -> int:
        return int(random.randint(0, len(cast(tuple[T], self._choices)) - 1))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Categorical:
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
    def __init__(
        self,
        min_value: float,
        max_value: float,
        log: bool = False,
        prior: float | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def has_prior(self) -> bool:
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> float:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return float(cast(float, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> float:
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
    def __init__(
        self,
        min_value: int,
        max_value: int,
        log: bool = False,
        prior: int | _Unset = _UNSET,
        prior_confidence: ConfidenceLevel | _Unset = _UNSET,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def min_value(self) -> int:
        return self._min_value

    @property
    def max_value(self) -> int:
        return self._max_value

    @property
    def has_prior(self) -> bool:
        return self._prior is not _UNSET and self._prior_confidence is not _UNSET

    @property
    def prior(self) -> int:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return int(cast(int, self._prior))

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        if not self.has_prior:
            raise ValueError("Domain has no prior defined.")
        return cast(ConfidenceLevel, self._prior_confidence)

    @property
    def range_compatibility_identifier(self) -> str:
        return f"{self._min_value}_{self._max_value}_{self._log}"

    def sample(self) -> int:
        if self._log:
            raise NotImplementedError("TODO.")
        return int(random.randint(self._min_value, self._max_value))

    def centered_around(
        self,
        center: int,
        confidence: ConfidenceLevel,
    ) -> Integer:
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
    def __init__(
        self,
        operator: Callable | str,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ):
        self._operator = operator
        self._args = tuple(args) if args else tuple()
        self._kwargs = kwargs if kwargs else {}

    @property
    def operator(self) -> Callable | str:
        return self._operator

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        return self._kwargs

    def get_attrs(self) -> Mapping[str, Any]:
        # TODO: simplify this. We know the fields. Maybe other places too.
        result: dict[str, Any] = {}
        for name, value in vars(self).items():
            name = name.lstrip("_")
            if isinstance(value, dict):
                for k, v in value.items():
                    # Multiple {{}} needed to escape surrounding '{' and '}'.
                    result[f"{name}{{{k}}}"] = v
            elif isinstance(value, tuple):
                for i, v in enumerate(value):
                    result[f"{name}[{i}]"] = v
            else:
                result[name] = value
        return result

    def from_attrs(self, attrs: Mapping[str, Any]) -> Operation:
        # TODO: simplify this. We know the fields. Maybe other places too.
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
    def __init__(self, source: Resolvable | str):
        self._source = source

    @property
    def source(self) -> Resolvable | str:
        return self._source

    @property
    def is_resampling_by_name(self) -> bool:
        return isinstance(self._source, str)

    def get_attrs(self) -> Mapping[str, Any]:
        if self.is_resampling_by_name:
            raise ValueError(f"This is a resampling by name, can't get attrs from it: {self.source!r}.")
        if not isinstance(self._source, Resolvable):
            raise ValueError(f"Source should be a resolvable object. Is: {self._source!r}.")
        return self._source.get_attrs()

    def from_attrs(self, attrs: Mapping[str, Any]) -> Resolvable:
        if self.is_resampling_by_name:
            raise ValueError(f"This is a resampling by name, can't create object for it: {self.source!r}.")
        if not isinstance(self._source, Resolvable):
            raise ValueError(f"Source should be a resolvable object. Is: {self._source!r}.")
        return self._source.from_attrs(attrs)


class Fidelity(Resolvable, Generic[T]):
    def __init__(self, domain: Integer | Float):
        if domain.has_prior:
            raise ValueError(f"The domain of a Fidelity can not have priors: {domain!r}.")
        self._domain = domain

    @property
    def min_value(self) -> int | float:
        return self._domain.min_value

    @property
    def max_value(self) -> int | float:
        return self._domain.max_value

    def get_attrs(self) -> Mapping[str, Any]:
        raise ValueError("For a Fidelity object there is nothing to resolve.")

    def from_attrs(self, attrs: Mapping[str, Any]) -> Fidelity:
        raise ValueError("For a Fidelity object there is nothing to resolve.")


# -------------------------------------------------


class OnlyPredefinedValuesSampler(DomainSampler):
    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        self._predefined_samplings = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        if current_path not in self._predefined_samplings:
            raise ValueError(f"No predefined value for path: {current_path!r}.")
        return cast(T, self._predefined_samplings[current_path])


class RandomSampler(DomainSampler):
    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        self._predefined_samplings = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        if current_path not in self._predefined_samplings:
            sampled_value = domain_obj.sample()
        else:
            sampled_value = cast(T, self._predefined_samplings[current_path])
        return sampled_value


class PriorOrFallbackSampler(DomainSampler):
    def __init__(
        self,
        fallback_sampler: DomainSampler,
        prior_use_probability: float,
    ):
        if not 0 <= prior_use_probability <= 1:
            raise ValueError(f"The given `prior_use_probability` value is out of range: {prior_use_probability!r}.")

        self._fallback_sampler = fallback_sampler
        self._prior_use_probability = prior_use_probability

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        use_prior = random.choices(
            (True, False),
            weights=(self._prior_use_probability, 1 - self._prior_use_probability),
            k=1,
        )[0]
        if domain_obj.has_prior and use_prior:
            return domain_obj.prior
        return self._fallback_sampler(
            domain_obj=domain_obj,
            current_path=current_path,
        )


class MutateByForgettingSampler(DomainSampler):
    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_forgets: int,
    ):
        if not isinstance(n_forgets, int) or n_forgets <= 0 or n_forgets > len(predefined_samplings):
            raise ValueError(f"Invalid value for `n_forgets`: {n_forgets!r}.")

        mutated_samplings_to_make = _mutate_samplings_to_make_by_forgetting(
            samplings_to_make=predefined_samplings,
            n_forgets=n_forgets,
        )

        self._random_sampler = RandomSampler(
            predefined_samplings=mutated_samplings_to_make,
        )

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)


class MutatateUsingCentersSampler(DomainSampler):
    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_mutations: int,
    ):
        if not isinstance(n_mutations, int) or n_mutations <= 0 or n_mutations > len(predefined_samplings):
            raise ValueError(f"Invalid value for `n_mutations`: {n_mutations!r}.")

        self._kept_samplings_to_make = _mutate_samplings_to_make_by_forgetting(
            samplings_to_make=predefined_samplings,
            n_forgets=n_mutations,
        )

        # Still remember the original choices. We'll use them as centers later.
        self._original_samplings_to_make = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        if current_path not in self._kept_samplings_to_make:
            # For this path we either have forgotten the value or we never had it.
            if current_path in self._original_samplings_to_make:
                # If we had a value for this path originally, use it as a center.
                original_value = self._original_samplings_to_make[current_path]
                sampled_value = domain_obj.centered_around(
                    center=original_value,
                    confidence=ConfidenceLevel.HIGH,
                ).sample()
            else:
                # We never had a value for this path, we can only sample from the domain.
                sampled_value = domain_obj.sample()
        else:
            # For this path we have chosen to keep the original value.
            sampled_value = cast(T, self._kept_samplings_to_make[current_path])

        return sampled_value


class CrossoverNotPossibleError(Exception):
    pass


class CrossoverByMixingSampler(DomainSampler):
    def __init__(
        self,
        predefined_samplings_1: Mapping[str, Any],
        predefined_samplings_2: Mapping[str, Any],
        prefer_first_probability: float,
    ):
        if not isinstance(prefer_first_probability, float) or not (0 <= prefer_first_probability <= 1):
            raise ValueError(f"Invalid value for `prefer_first_probability`: {prefer_first_probability!r}.")

        (
            made_any_crossovers,
            crossed_over_samplings_to_make,
        ) = _crossover_samplings_to_make_by_mixing(
            predefined_samplings_1=predefined_samplings_1,
            predefined_samplings_2=predefined_samplings_2,
            prefer_first_probability=prefer_first_probability,
        )

        if not made_any_crossovers:
            raise CrossoverNotPossibleError("No crossovers were made.")

        self._random_sampler = RandomSampler(
            predefined_samplings=crossed_over_samplings_to_make,
        )

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)


def _mutate_samplings_to_make_by_forgetting(
    samplings_to_make: Mapping[str, Any],
    n_forgets: int,
) -> Mapping[str, Any]:
    mutated_samplings_to_make = dict(**samplings_to_make)

    samplings_to_delete = random.sample(
        list(samplings_to_make.keys()),
        k=n_forgets,
    )

    for choice_to_delete in samplings_to_delete:
        mutated_samplings_to_make.pop(choice_to_delete)

    return mutated_samplings_to_make


def _crossover_samplings_to_make_by_mixing(
    predefined_samplings_1: Mapping[str, Any],
    predefined_samplings_2: Mapping[str, Any],
    prefer_first_probability: float,
) -> tuple[bool, Mapping[str, Any]]:
    crossed_over_samplings = dict(**predefined_samplings_1)
    made_any_crossovers = False

    for path, sampled_value_in_2 in predefined_samplings_2.items():
        if path in crossed_over_samplings:
            use_value_from_2 = random.choices(
                (False, True),
                weights=(prefer_first_probability, 1 - prefer_first_probability),
                k=1,
            )[0]
            if use_value_from_2:
                crossed_over_samplings[path] = sampled_value_in_2
                made_any_crossovers = True
        else:
            crossed_over_samplings[path] = sampled_value_in_2

    return made_any_crossovers, crossed_over_samplings


# -------------------------------------------------


class SamplingResolutionContext:
    def __init__(
        self,
        *,
        resolution_root: Resolvable,
        domain_sampler: DomainSampler,
        environment_values: Mapping[str, Any],
    ):
        if not isinstance(resolution_root, Resolvable):
            raise ValueError(f"The received `resolution_root` is not a Resolvable: {resolution_root!r}.")

        if not isinstance(domain_sampler, DomainSampler):
            raise ValueError(f"The received `domain_sampler` is not a DomainSampler: {domain_sampler!r}.")

        if not isinstance(environment_values, Mapping):
            raise ValueError(f"The received `environment_values` is not a Mapping: {environment_values!r}.")

        # `_resolution_root` stores the root of the resolution.
        self._resolution_root: Resolvable = resolution_root

        # `_domain_sampler` stores the object responsible for sampling from Domain objects.
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
        return self._resolution_root

    @property
    def samplings_made(self) -> Mapping[str, Any]:
        return self._samplings_made

    @property
    def environment_values(self) -> Mapping[str, Any]:
        return self._environment_values

    @contextlib.contextmanager
    def resolving(self, _obj: Any, name: str) -> Generator[None]:
        if not name or not isinstance(name, str):
            raise ValueError(f"Given name for what we are resolving is invalid: {name!r}.")

        # It is possible that the received object has already been resolved.
        # That is expected and is okay, so no check is made for it.
        # For example, in the case of a Resampled we can receive the same object again.

        self._current_path_parts.append(name)
        try:
            yield
        finally:
            self._current_path_parts.pop()

    def was_already_resolved(self, obj: Any) -> bool:
        return obj in self._resolved_objects

    def add_resolved(self, original: Any, resolved: Any) -> None:
        if self.was_already_resolved(original):
            raise ValueError(
                f"Original object has already been resolved: {original!r}. "
                + "\nIf you are doing resampling by name, "
                + "make sure you are not forgetting to request resampling also for related objects."
                + "\nOtherwise it could lead to infinite recursion."
            )
        if isinstance(original, Resampled):
            raise ValueError(f"Attempting to add a Resampled object to resolved values: {original!r}.")
        self._resolved_objects[original] = resolved

    def get_resolved(self, obj: Any) -> Any:
        try:
            return self._resolved_objects[obj]
        except KeyError:
            raise ValueError(f"Given object was not already resolved. Please check first: {obj!r}")

    def sample_from(self, domain_obj: Domain) -> Any:
        # Each `domain_obj` is only ever sampled from once.
        # This is okay and the expected behavior.
        # For each `domain_obj`, its sampled value is either directly stored itself,
        # or is used in some other Resolvable.
        # In both cases that sampled value is cached for later uses,
        # and so the `domain_obj` will not be re-sampled from again.
        if self.was_already_resolved(domain_obj):
            raise ValueError(
                f"We have already sampled a value for the given domain object: {domain_obj!r}."
                + "\nThis should not be happening."
            )

        # The range compatibility identifier is there to make sure when we say
        # the path matches, that the range for the value we are looking up also matches.
        domain_obj_type_name = type(domain_obj).__name__.lower()
        range_compatibility_identifier = domain_obj.range_compatibility_identifier
        domain_obj_identifier = f"{domain_obj_type_name}__{range_compatibility_identifier}"

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
        try:
            return self._environment_values[var_name]
        except KeyError:
            raise ValueError(f"No value is available for the environment variable {var_name!r}.")


class SamplingResolver:
    def __call__(
        self,
        obj: Resolvable,
        domain_sampler: DomainSampler,
        environment_values: Mapping[str, Any],
    ) -> tuple[Resolvable, SamplingResolutionContext]:
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
                f"The default resolver is not supposed to be called for resolvable objects. Received: {any_obj!r}."
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
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

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
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

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
                    resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
                else:
                    resolved_attr_value = initial_attr_value
            else:
                resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        resolved_categorical_obj = categorical_obj
        if needed_resolving:
            resolved_categorical_obj = cast(Categorical, categorical_obj.from_attrs(final_attrs))

        try:
            sampled_index = context.sample_from(resolved_categorical_obj)
        except Exception as e:
            raise ValueError(f"Failed to sample from {resolved_categorical_obj!r}.") from e
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

        initial_attrs = operation_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        result = operation_obj
        if needed_resolving:
            result = operation_obj.from_attrs(final_attrs)

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
        result = self._resolve(resolvable_to_resample_obj, f"resampled_{type_name}", context)

        return result

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

        names_for_this_fidelity_obj = list(
            attr_name
            for attr_name, attr_value in context.resolution_root.get_attrs().items()
            if attr_value is fidelity_obj
        )

        if len(names_for_this_fidelity_obj) == 0:
            raise ValueError("A fidelity object should be a direct attribute of the pipeline.")
        elif len(names_for_this_fidelity_obj) > 1:
            raise ValueError("A fidelity object should only be referenced once in the pipeline.")

        fidelity_name = names_for_this_fidelity_obj[0]

        try:
            result = context.get_value_from_environment(fidelity_name)
        except ValueError:
            raise ValueError(f"No value is available in the environment for fidelity {fidelity_name!r}.")

        if not fidelity_obj.min_value <= result <= fidelity_obj.max_value:
            raise ValueError(
                f"Value for fidelity with name {fidelity_name!r} is outside its allowed range "
                + f"[{fidelity_obj.min_value!r}, {fidelity_obj.max_value!r}]. "
                + f"Received: {result!r}."
            )

        context.add_resolved(fidelity_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: Resolvable,
        context: SamplingResolutionContext,
    ) -> Any:
        # Called when no specialized resolver was available for the specific resolvable type.
        # That is not something that is normally expected.
        raise ValueError(f"No specialized resolver was registered for object of type {type(resolvable_obj)!r}.")


def resolve(
    pipeline: P,
    domain_sampler: DomainSampler | None = None,
    environment_values: Mapping[str, Any] | None = None,
) -> tuple[P, SamplingResolutionContext]:
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
    operator = cast(Callable, operation.operator)

    operation_args = []
    for arg in operation.args:
        if isinstance(arg, Operation):
            arg = convert_operation_to_callable(arg)
        operation_args.append(arg)

    operation_kwargs = {}
    for kwarg_name, kwarg_value in operation.kwargs.items():
        if isinstance(kwarg_value, Operation):
            kwarg_value = convert_operation_to_callable(kwarg_value)
        operation_kwargs[kwarg_name] = kwarg_value

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
    unwrapped_config = tuple(_operation_to_unwrapped_config(operation))
    return cast(str, config_string.wrap_config_into_string(unwrapped_config))


# -------------------------------------------------


class RandomSearch:
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

        self._environment_values = {}
        fidelity_attrs = self._pipeline.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            self._environment_values[fidelity_name] = fidelity_obj.max_value

        self._random_sampler = RandomSampler(predefined_samplings={})

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
        n_prev_trials = len(trials)
        n_requested = 1 if n is None else n
        return_single = n is None

        chosen_pipelines = [
            resolve(
                pipeline=self._pipeline,
                domain_sampler=self._random_sampler,
                environment_values=self._environment_values,
            )
            for _ in range(n_requested)
        ]

        return _prepare_sampled_configs(chosen_pipelines, n_prev_trials, return_single)


class ComplexRandomSearch:
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

        self._environment_values = {}
        fidelity_attrs = self._pipeline.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            self._environment_values[fidelity_name] = fidelity_obj.max_value

        self._random_sampler = RandomSampler(
            predefined_samplings={},
        )
        self._try_always_priors_sampler = PriorOrFallbackSampler(
            fallback_sampler=self._random_sampler,
            prior_use_probability=1,
        )
        self._sometimes_priors_sampler = PriorOrFallbackSampler(
            fallback_sampler=self._random_sampler,
            prior_use_probability=0.1,
        )

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
        n_prev_trials = len(trials)
        n_requested = 1 if n is None else n
        return_single = n is None

        random_pipelines = [
            resolve(
                pipeline=self._pipeline,
                domain_sampler=self._random_sampler,
                environment_values=self._environment_values,
            )
            for _ in range(n_requested * 5)
        ]
        sometimes_priors_pipelines = [
            resolve(
                pipeline=self._pipeline,
                domain_sampler=self._sometimes_priors_sampler,
                environment_values=self._environment_values,
            )
            for _ in range(n_requested * 5)
        ]

        mutated_incumbents = []
        crossed_over_incumbents = []

        successful_trials = list(
            filter(
                lambda trial: trial.report.reported_as == trial.State.SUCCESS,
                trials.values(),
            )
        )
        if len(successful_trials) > 0:
            n_top_trials = 5
            top_trials = heapq.nsmallest(
                n_top_trials,
                successful_trials,
                key=lambda trial: trial.report.objective_to_minimize,
            )  # Will have up to `n_top_trials` items.

            # Do some mutations.
            for top_trial in top_trials:
                top_trial_config = top_trial.config

                # Mutate by resampling around some values of the original config.
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutatateUsingCentersSampler(
                            predefined_samplings=top_trial_config,
                            n_mutations=1,
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutatateUsingCentersSampler(
                            predefined_samplings=top_trial_config,
                            n_mutations=random.randint(1, int(len(top_trial_config) / 2)),
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]

                # Mutate by completely forgetting some values of the original config.
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutateByForgettingSampler(
                            predefined_samplings=top_trial_config,
                            n_forgets=1,
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutateByForgettingSampler(
                            predefined_samplings=top_trial_config,
                            n_forgets=random.randint(1, int(len(top_trial_config) / 2)),
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]

            # Do some crossovers.
            if len(top_trials) > 1:
                for _ in range(n_requested * 3):
                    trial_1, trial_2 = random.sample(top_trials, k=2)

                    try:
                        crossover_sampler = CrossoverByMixingSampler(
                            predefined_samplings_1=trial_1.config,
                            predefined_samplings_2=trial_2.config,
                            prefer_first_probability=0.5,
                        )
                    except CrossoverNotPossibleError:
                        # A crossover was not possible for them. Do nothing.
                        pass
                    else:
                        crossed_over_incumbents.append(
                            resolve(
                                pipeline=self._pipeline,
                                domain_sampler=crossover_sampler,
                                environment_values=self._environment_values,
                            ),
                        )

                    try:
                        crossover_sampler = CrossoverByMixingSampler(
                            predefined_samplings_1=trial_2.config,
                            predefined_samplings_2=trial_1.config,
                            prefer_first_probability=0.5,
                        )
                    except CrossoverNotPossibleError:
                        # A crossover was not possible for them. Do nothing.
                        pass
                    else:
                        crossed_over_incumbents.append(
                            resolve(
                                pipeline=self._pipeline,
                                domain_sampler=crossover_sampler,
                                environment_values=self._environment_values,
                            ),
                        )

        all_sampled_pipelines = [
            *random_pipelines,
            *sometimes_priors_pipelines,
            *mutated_incumbents,
            *crossed_over_incumbents,
        ]

        # Here we can have a model which picks from all the sampled pipelines.
        # Currently, we just pick randomly from them.
        chosen_pipelines = random.sample(all_sampled_pipelines, k=n_requested)

        if n_prev_trials == 0:
            # In this case, always include the prior pipeline.
            prior_pipeline = resolve(
                pipeline=self._pipeline,
                domain_sampler=self._try_always_priors_sampler,
                environment_values=self._environment_values,
            )
            chosen_pipelines[0] = prior_pipeline

        return _prepare_sampled_configs(chosen_pipelines, n_prev_trials, return_single)


# -------------------------------------------------


class _NepsCompatConverter:
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
    return_single: bool,
) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
    configs = []
    for i, (_resolved_pipeline, resolution_context) in enumerate(chosen_pipelines):
        neps_config = _NepsCompatConverter.to_neps_config(
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


def adjust_evaluation_pipeline_for_new_space(
    evaluation_pipeline: Callable,
    pipeline_space: P,
    operation_converter: Callable[[Operation], Any] = convert_operation_to_callable,
) -> Callable | str:
    @functools.wraps(evaluation_pipeline)
    def inner(*args: Any, **kwargs: Any) -> Any:
        # `kwargs` can contain other things not related to
        # the samplings to make or to environment values.
        # That is not an issue. Those items will be passed through.

        sampled_pipeline_data = _NepsCompatConverter.from_neps_config(config=kwargs)

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

        # So that we still pass the kwargs not related to the config.
        # Take away all the kwargs which were related to samplings made.
        new_kwargs = dict(**sampled_pipeline_data.extra_kwargs)
        # Then add all the kwargs from the config.
        new_kwargs.update(config)

        return evaluation_pipeline(*args, **new_kwargs)

    return inner
