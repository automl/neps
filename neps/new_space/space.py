from __future__ import annotations

import abc
import random
import math
import enum
from contextlib import contextmanager
from typing import TypeVar, Generic, Sequence, Any, Protocol, runtime_checkable, cast

from neps.new_space import config_string


T = TypeVar("T")
P = TypeVar("P", bound="Pipeline")


# -------------------------------------------------


@runtime_checkable
class Resolvable(Protocol):
    def get_attrs(self) -> dict[str, Any]:
        raise NotImplementedError()

    def from_attrs(self, attrs: dict[str, Any]) -> Resolvable:
        raise NotImplementedError()


@runtime_checkable
class Resolver(Protocol):
    def resolve(self, obj: Resolvable) -> Any:
        raise NotImplementedError()


# -------------------------------------------------


class Pipeline(Resolvable):
    def get_attrs(self) -> dict[str, Any]:
        attrs = {}

        for attr_name, attr_value in vars(self.__class__).items():
            if attr_name.startswith("_") or callable(attr_value):
                continue
            attrs[attr_name] = attr_value

        for attr_name, attr_value in vars(self).items():
            if attr_name.startswith("_") or callable(attr_value):
                continue
            attrs[attr_name] = attr_value

        return attrs

    def from_attrs(self, attrs: dict[str, Any]) -> Pipeline:
        new_pipeline = Pipeline()
        for name, value in attrs.items():
            setattr(new_pipeline, name, value)
        return new_pipeline


class ConfidenceLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# TODO: the Domains need to be adjusted so that they return values (or not?)
# TODO: move the sampling out of the Domains. They just describe domains. Or not?
class Domain(Resolvable, abc.ABC, Generic[T]):
    @property
    @abc.abstractmethod
    def prior(self) -> T:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prior_confidence(self) -> ConfidenceLevel:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> T:
        raise NotImplementedError()

    def get_attrs(self) -> dict[str, Any]:
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def from_attrs(self, attrs: dict[str, Any]) -> Domain[T]:
        return type(self)(**attrs)

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"


class Categorical(Domain[int], Generic[T]):
    def __init__(
        self,
        choices: tuple[T | Domain[T] | Resolvable, ...] | Domain[T],
        prior_index: int | Domain[int],
        prior_confidence: ConfidenceLevel,
    ):
        self._choices: tuple[T | Domain[T] | Resolvable, ...] | Domain[T]
        if isinstance(choices, Sequence):
            self._choices = tuple(choice for choice in choices)
        else:
            self._choices = choices
        self._prior_index = prior_index
        self._prior_confidence = prior_confidence

    @property
    def choices(self) -> tuple[T | Domain[T] | Resolvable, ...] | Domain[T]:
        return self._choices

    @property
    def prior(self) -> int:
        self._check_fully_resolved()
        return cast(int, self._prior_index)

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        return self._prior_confidence

    def sample(self) -> int:
        self._check_fully_resolved()
        return random.randint(0, len(cast(tuple[T], self._choices)) - 1)

    def _check_fully_resolved(self) -> None:
        if not isinstance(self._choices, tuple):
            raise ValueError(f"Choices have not been resolved: {self._choices}")
        if not isinstance(self._prior_index, int):
            raise ValueError(f"Default index has not been resolved: {self._prior_index}")


class Float(Domain[float]):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        log: bool,
        prior: float,
        prior_confidence: ConfidenceLevel,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._log = log
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def prior(self) -> float:
        return self._prior

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        return self._prior_confidence

    def sample(self) -> float:
        if self._log:
            log_min = math.log(self._min_value)
            log_max = math.log(self._max_value)
            return math.exp(random.uniform(log_min, log_max))
        return random.uniform(self._min_value, self._max_value)


class Integer(Domain[int]):
    def __init__(
        self,
        min_value: int,
        max_value: int,
        prior: int,
        prior_confidence: ConfidenceLevel,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._prior = prior
        self._prior_confidence = prior_confidence

    @property
    def prior(self) -> int:
        return self._prior

    @property
    def prior_confidence(self) -> ConfidenceLevel:
        return self._prior_confidence

    def sample(self) -> int:
        return random.randint(self._min_value, self._max_value)


class Resampled(Resolvable, Generic[T]):
    def __init__(self, source: T | Domain[T] | Resolvable):
        self._source = source

    def get_attrs(self) -> dict[str, Any]:
        if not isinstance(self._source, Resolvable):
            raise ValueError(f"Source should be a resolvable object. Is: {self._source!r}")
        return self._source.get_attrs()

    def from_attrs(self, attrs: dict[str, Any]) -> Resolvable:
        if not isinstance(self._source, Resolvable):
            raise ValueError(f"Source should be a resolvable object. Is: {self._source!r}")
        return self._source.from_attrs(attrs)


class Operation(Resolvable):
    def __init__(
        self,
        operator: Any,
        kwargs: dict[str, Any] | None = None,
        args: Sequence[Any] | None = None,
    ):
        self._operator = operator
        self._kwargs = kwargs if kwargs else {}
        self._args = tuple(args) if args else tuple()

    @property
    def operator(self) -> Any:
        return self._operator

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._kwargs

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    def get_attrs(self) -> dict[str, Any]:
        # todo: simplify this. We know the fields. Maybe other places too
        result: dict[str, Any] = {}
        for name, value in vars(self).items():
            name = name.lstrip("_")
            if isinstance(value, dict):
                for k, v in value.items():
                    result[f"{name}__{k}"] = v
            elif isinstance(value, tuple):
                for i, v in enumerate(value):
                    result[f"{name}[{i}]"] = v
            else:
                result[name] = value
        return result

    def from_attrs(self, attrs: dict[str, Any]) -> Operation:
        # todo: simplify this. We know the fields. Maybe other places too
        final_attrs: dict[str, Any] = {}
        for name, value in attrs.items():
            if "__" in name:
                base, key = name.split("__")
                final_attrs.setdefault(base, {})[key] = value
            elif "[" in name and "]" in name:
                base, idx_str = name.split("[")
                idx = int(idx_str.rstrip("]"))
                final_attrs.setdefault(base, []).insert(idx, value)
            else:
                final_attrs[name] = value
        return type(self)(**final_attrs)

    def __str__(self) -> str:
        kwarg_names = []
        for k, v in self._kwargs.items():
            kwarg_names.append(f"{k}={v}")
        kwarg_names_str = ",".join(kwarg_names)

        arg_names = []
        for op in self._args:
            arg_names.append(str(op))
        arg_names_str = ",".join(arg_names)

        return f"{self._operator}({kwarg_names_str};{arg_names_str})"


class Value(Resolvable, Generic[T]):
    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    def get_attrs(self) -> dict[str, Any]:
        raise NotImplementedError()

    def from_attrs(self, attrs: dict[str, Any]) -> Resolvable:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self._value}"


# -------------------------------------------------


class ResolutionContext:
    def __init__(self, samplings_to_make: dict[str, Any] | None = None):
        self._sampled_values: dict[str, Any]
        if samplings_to_make:
            self._sampled_values = {k: v for k, v in samplings_to_make.items()}
        else:
            self._sampled_values = {}
        self._resolved_objects: dict[Any, Any] = {}
        self._current_path: list[str] = []

    @contextmanager
    def resolving(self, obj: Any, name: str):
        added = False
        if name:
            self._current_path.append(name)
            added = True
        try:
            yield
        finally:
            if added:
                self._current_path.pop()

    def add_resolved(self, original: Any, resolved: Any) -> None:
        if isinstance(original, Resampled):
            raise ValueError(f"Attempting to add Resampled object to resolved values: {original}")
        self._resolved_objects[original] = resolved

    def get_resolved(self, obj: Any) -> Any | None:
        return self._resolved_objects.get(obj)

    def __contains__(self, item):
        return item in self._resolved_objects

    def sample_from(self, domain_obj: Domain) -> Any:
        # Each domain object is only ever sampled from once.
        # This is okay and the expected behavior.
        # For each domain, its result is either stored as a value,
        # or used in some other Resolvable, which is cached for later uses,
        # and so will not need to be re-sampled again.

        # The field `_sampled_values` stores what we have sampled
        # and can be used later in case we want to redo a resolving.
        # Starting with the same pipeline space, iterating that space
        # in a consistent order and making the same samplings along the way,
        # means we will end up with the same final configuration.

        joined_current_path = ".".join(self._current_path)
        joined_current_path += f".sampled_{type(domain_obj).__name__.lower()}"

        if joined_current_path in self._sampled_values:
            # we already have a predefined value to return for this domain
            return self._sampled_values[joined_current_path]

        sampled_value = domain_obj.sample()
        self._sampled_values[joined_current_path] = sampled_value

        return sampled_value

    def get_sampled_values(self):
        return {k: v for k, v in self._sampled_values.items()}

    def get_stats(self) -> dict[str, Any]:
        return {
            "size": len(self._resolved_objects),
            "sampled_values": self._sampled_values,
        }


class DefaultResolver(Resolver):
    def __init__(self, context: ResolutionContext | None = None):
        self._context = context or ResolutionContext()

    def resolve(self, pipeline: P | Resolvable) -> tuple[P | Resolvable, ResolutionContext]:
        return self._resolve(pipeline, ""), self._context

    def _resolve(self, obj: Any, name: str) -> Any:
        with self._context.resolving(obj, name):
            if not isinstance(obj, Resolvable):
                return self._resolve_other(obj)
            if isinstance(obj, Pipeline):
                return self._resolve_pipeline(obj)
            elif isinstance(obj, Categorical):
                return self._resolve_categorical(obj)
            elif isinstance(obj, Domain):
                return self._resolve_domain(obj)
            elif isinstance(obj, Operation):
                return self._resolve_operation(obj)
            elif isinstance(obj, Resampled):
                return self._resolve_resampled(obj)

        raise ValueError(f"This part should not be reachable. Received: {obj}, name={name}")

    def _resolve_pipeline(self, pipeline_obj: Pipeline) -> Any:
        if pipeline_obj in self._context:
            return self._context.get_resolved(pipeline_obj)

        initial_attrs = pipeline_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        result = pipeline_obj
        if needed_resolving:
            result = cast(Pipeline, pipeline_obj).from_attrs(final_attrs)

        self._context.add_resolved(pipeline_obj, result)

        return result

    def _resolve_domain(self, domain_obj: Domain[Any]) -> Any:
        if domain_obj in self._context:
            return self._context.get_resolved(domain_obj)

        initial_attrs = domain_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        resolved_domain_obj = domain_obj
        if needed_resolving:
            resolved_domain_obj = domain_obj.from_attrs(final_attrs)

        try:
            sampled_value = self._context.sample_from(resolved_domain_obj)
        except Exception as e:
            raise ValueError(f"Failed to sample from {resolved_domain_obj}") from e
        result = self._resolve(sampled_value, "sampled_value")

        self._context.add_resolved(resolved_domain_obj, result)
        self._context.add_resolved(domain_obj, result)

        return result

    def _resolve_categorical(self, categorical_obj: Categorical[Any]) -> Any:
        if categorical_obj in self._context:
            return self._context.get_resolved(categorical_obj)

        # in the case of categorical choices, we may skip resolving each choice initially,
        # only after sampling we go into resolving whatever choice was chosen.
        # This avoids resolving things which won't be needed at all.
        # If the choices themselves come from some Resolvable, they will be resolved.

        initial_attrs = categorical_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            if attr_name == "choices":
                if isinstance(initial_attr_value, Resolvable):
                    # resolving here like below works fine since the expectation
                    # is that we will get back a tuple of choices.
                    # Any element in that tuple can be a Resolvable,
                    # but will not be resolved from the call directly below,
                    # as the tuple is returned as is,
                    # without going into resolving its elements.
                    # If we add a `_resolve_tuple` functionality to go into tuples
                    # and resolve their contents, the call below will likely
                    # lead to too much work being done or issues.
                    resolved_attr_value = self._resolve(initial_attr_value, attr_name)
                else:
                    resolved_attr_value = initial_attr_value
            else:
                resolved_attr_value = self._resolve(initial_attr_value, attr_name)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        resolved_categorical_obj = categorical_obj
        if needed_resolving:
            resolved_categorical_obj = cast(Categorical, categorical_obj.from_attrs(final_attrs))

        try:
            sampled_index_value = self._context.sample_from(resolved_categorical_obj)
        except Exception as e:
            raise ValueError(f"Failed to sample from {resolved_categorical_obj}") from e
        sampled_value = cast(tuple, resolved_categorical_obj.choices)[sampled_index_value]
        result = self._resolve(sampled_value, "sampled_value")

        self._context.add_resolved(resolved_categorical_obj, result)
        self._context.add_resolved(categorical_obj, result)

        return result

    def _resolve_operation(self, operation_obj: Operation) -> Any:
        if operation_obj in self._context:
            return self._context.get_resolved(operation_obj)

        initial_attrs = operation_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name)
            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (initial_attr_value is not resolved_attr_value)

        result = operation_obj
        if needed_resolving:
            result = operation_obj.from_attrs(final_attrs)

        self._context.add_resolved(operation_obj, result)

        return result

    def _resolve_resampled(self, resampled_obj: Resampled) -> Any:
        # the results of Resampled are never stored or looked up from cache
        # since it would break the logic of their expected behavior.
        # Particularly, when Resampled objects are nested (at any depth) inside of
        # other Resampled objects, adding them to the resolution context would result
        # in the resolution not doing the right thing
        initial_attrs = resampled_obj.get_attrs()
        resolvable_to_resample_obj = cast(Resampled, resampled_obj).from_attrs(initial_attrs)
        # we mark that we are entering an object to resample
        result = self._resolve(
            resolvable_to_resample_obj,
            f"resampled_{type(resolvable_to_resample_obj).__name__.lower()}",
        )
        return result

    def _resolve_other(self, obj: Any) -> Any:
        # no need to store or lookup from context
        return obj


def resolve(
    pipeline: P,
    samplings_to_make: dict[str, Any] | None = None,
) -> tuple[P, dict[str, Any]]:
    if samplings_to_make:
        context = ResolutionContext(samplings_to_make)
    else:
        context = None
    resolver = DefaultResolver(context)
    resolved_pipeline, context = resolver.resolve(pipeline)
    return cast(P, resolved_pipeline), context.get_sampled_values()


# -------------------------------------------------


def _to_unwrapped_config(
    operation: Operation | str,
    context: list[config_string.UnwrappedConfigStringPart],
    level: int = 1,
) -> tuple[config_string.UnwrappedConfigStringPart, ...]:
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
        context.append(item)
        for operand in operation.args:
            _to_unwrapped_config(operand, context, level + 1)
    else:
        item = config_string.UnwrappedConfigStringPart(
            level=level,
            opening_index=-1,
            operator=operation,
            hyperparameters="",
            operands="",
        )
        context.append(item)
    return tuple(context)


def to_config_string(operation: Operation) -> str:
    context: list[config_string.UnwrappedConfigStringPart] = []
    unwrapped_config = _to_unwrapped_config(operation, context)
    result = config_string.wrap_config_into_string(unwrapped_config)
    return result
