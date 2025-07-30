"""This module provides functionality for resolving NePS spaces, including sampling from
domains, resolving pipelines, and handling various resolvable objects.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Callable, Generator, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeVar, cast

import neps
from neps.optimizers import algorithms, optimizer
from neps.space.neps_spaces import config_string
from neps.space.neps_spaces.parameters import (
    Categorical,
    Domain,
    Fidelity,
    Float,
    Integer,
    Operation,
    PipelineSpace,
    Resampled,
    Resolvable,
)
from neps.space.neps_spaces.sampling import (
    DomainSampler,
    OnlyPredefinedValuesSampler,
    RandomSampler,
)
from neps.space.parsing import convert_mapping

if TYPE_CHECKING:
    from neps.space import SearchSpace

P = TypeVar("P", bound="PipelineSpace")


class SamplingResolutionContext:
    """A context for resolving samplings in a NePS space.
    It manages the resolution root, domain sampler, environment values,
    and keeps track of samplings made and resolved objects.

    Args:
        resolution_root: The root of the resolution, which should be a Resolvable
            object.
        domain_sampler: The DomainSampler to use for sampling from Domain objects.
        environment_values: A mapping of environment values that are fixed and not
            related to samplings. These values can be used in the resolution process.

    Raises:
        ValueError: If the resolution_root is not a Resolvable, or if the domain_sampler
            is not a DomainSampler, or if the environment_values is not a Mapping.
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

        Args:
            resolution_root: The root of the resolution, which should be a Resolvable
                object.
            domain_sampler: The DomainSampler to use for sampling from Domain objects.
            environment_values: A mapping of environment values that are fixed and not
                related to samplings. These values can be used in the resolution process.

        Raises:
            ValueError: If the resolution_root is not a Resolvable, or if the
                domain_sampler is not a DomainSampler, or if the environment_values is
                not a Mapping.
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

        Returns:
            The root of the resolution, which should be a Resolvable object.
        """
        return self._resolution_root

    @property
    def samplings_made(self) -> Mapping[str, Any]:
        """Get the samplings made during the resolution process.

        Returns:
            A mapping of paths to sampled values.
        """
        return self._samplings_made

    @property
    def environment_values(self) -> Mapping[str, Any]:
        """Get the environment values that are fixed and not related to samplings.

        Returns:
            A mapping of environment variable names to their values.
        """
        return self._environment_values

    @contextlib.contextmanager
    def resolving(self, _obj: Any, name: str) -> Generator[None]:
        """Context manager for resolving an object in the current resolution context.

        Args:
            _obj: The object being resolved, can be any type.
            name: The name of the object being resolved, used for debugging.

        Raises:
            ValueError: If the name is not a valid string.
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

        Args:
            obj: The object to check if it was already resolved.

        Returns:
            True if the object was already resolved, False otherwise.
        """
        return obj in self._resolved_objects

    def add_resolved(self, original: Any, resolved: Any) -> None:
        """Add a resolved object to the context.

        Args:
            original: The original object that was resolved.
            resolved: The resolved value of the original object.

        Raises:
            ValueError: If the original object was already resolved or if it is a
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

        Args:
            obj: The object for which to get the resolved value.

        Returns:
            The resolved value of the object.

        Raises:
            ValueError: If the object was not already resolved in the context.
        """
        try:
            return self._resolved_objects[obj]
        except KeyError as err:
            raise ValueError(
                f"Given object was not already resolved. Please check first: {obj!r}"
            ) from err

    def sample_from(self, domain_obj: Domain) -> Any:
        """Sample a value from the given domain object.

        Args:
            domain_obj: The domain object from which to sample a value.

        Returns:
            The sampled value from the domain object.

        Raises:
            ValueError: If the domain object was already resolved or if the path
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

        Args:
            var_name: The name of the environment variable to get the value from.

        Returns:
            The value of the environment variable.

        Raises:
            ValueError: If the environment variable is not found in the context.
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
    """

    def __call__(
        self,
        obj: Resolvable,
        domain_sampler: DomainSampler,
        environment_values: Mapping[str, Any],
    ) -> tuple[Resolvable, SamplingResolutionContext]:
        """Resolve the given object in the context of the provided domain sampler and
        environment values.

        Args:
            obj: The Resolvable object to resolve.
            domain_sampler: The DomainSampler to use for sampling from Domain objects.
            environment_values: A mapping of environment values that are fixed and not
                related to samplings.

        Returns:
            A tuple containing the resolved object and the
                SamplingResolutionContext.

        Raises:
            ValueError: If the object is not a Resolvable, or if the domain_sampler
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
        pipeline_obj: PipelineSpace,
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
        resolvable_obj: tuple,
        context: SamplingResolutionContext,  # noqa: ARG002
    ) -> Any:
        return tuple(self._resolve_collection(resolvable_obj, context))

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: list,
        context: SamplingResolutionContext,  # noqa: ARG002
    ) -> Any:
        return self._resolve_collection(resolvable_obj, context)

    def _resolve_collection(
        self,
        resolvable_obj: tuple | list,
        context: SamplingResolutionContext,  # noqa: ARG002
    ) -> list[Any]:
        result = []
        for idx, item in enumerate(resolvable_obj):
            result.append(self._resolve(item, f"collection[{idx}]", context))
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

    Args:
        pipeline: The pipeline to resolve, which should be a Pipeline object.
        domain_sampler: The DomainSampler to use for sampling from Domain objects.
            If None, a RandomSampler with no predefined values will be used.
        environment_values: A mapping of environment variable names to their values.
            If None, an empty mapping will be used.

    Returns:
        A tuple containing the resolved pipeline and the SamplingResolutionContext.

    Raises:
        ValueError: If the pipeline is not a Pipeline object or if the domain_sampler
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

    Args:
        operation: The Operation to convert.

    Returns:
        A callable that represents the operation.

    Raises:
        ValueError: If the operation is not a valid Operation object.
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

    Args:
        operation: The Operation to convert.

    Returns:
        A string representation of the operation.

    Raises:
        ValueError: If the operation is not a valid Operation object.
    """
    unwrapped_config = tuple(_operation_to_unwrapped_config(operation))
    return config_string.wrap_config_into_string(unwrapped_config)


# -------------------------------------------------


class NepsCompatConverter:
    """A class to convert between NePS configurations and NEPS-compatible configurations.
    It provides methods to convert a SamplingResolutionContext to a NEPS-compatible config
    and to convert a NEPS-compatible config back to a SamplingResolutionContext.
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

        Args:
            resolution_context: The SamplingResolutionContext to convert.

        Returns:
            A mapping of NEPS-compatible configuration keys to their values.

        Raises:
            ValueError: If the resolution_context is not a SamplingResolutionContext.
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

        Args:
            config: A mapping of NEPS-compatible configuration keys to their values.

        Returns:
            A _FromNepsConfigResult containing predefined samplings,
                environment values, and extra kwargs.

        Raises:
            ValueError: If the config is not a valid NEPS-compatible config.
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
    chosen_pipelines: list[tuple[PipelineSpace, SamplingResolutionContext]],
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

    Args:
        evaluation_pipeline: The evaluation pipeline to adjust.
        pipeline_space: The NePS pipeline space to sample from.
        operation_converter: A callable to convert Operation objects to a format
            compatible with the evaluation pipeline.

    Returns:
        A wrapped evaluation pipeline that samples from the NePS space.

    Raises:
        ValueError: If the evaluation_pipeline is not callable or if the
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
                # If the operator is a not a string, we convert it to a callable.
                if isinstance(value.operator, str):
                    config[name] = value.operator
                else:
                    config[name] = operation_converter(value)

        # So that we still pass the kwargs not related to the config,
        # start with the extra kwargs we passed to the converter.
        new_kwargs = dict(**sampled_pipeline_data.extra_kwargs)
        # Then add all the kwargs from the config.
        new_kwargs.update(config)

        return evaluation_pipeline(*args, **new_kwargs)

    return inner


def convert_neps_to_classic_search_space(space: PipelineSpace) -> SearchSpace | None:
    """Convert a NePS space to a classic SearchSpace if possible.
    This function checks if the NePS space can be converted to a classic SearchSpace
    by ensuring that it does not contain any complex types like Operation or Resampled,
    and that all choices of Categorical parameters are of basic types (int, str, float).
    If the checks pass, it converts the NePS space to a classic SearchSpace.

    Args:
        space: The NePS space to convert, which should be a Pipeline object.

    Returns:
        A classic SearchSpace if the conversion is possible, otherwise None.
    """
    # First check: No parameters are of type Operation or Resampled
    if not any(
        isinstance(param, Operation | Resampled) for param in space.get_attrs().values()
    ):
        # Second check: All choices of all categoricals are of basic
        # types i.e. int, str or float
        categoricals = [
            param
            for param in space.get_attrs().values()
            if isinstance(param, Categorical)
        ]
        if all(
            any(
                all(isinstance(choice, datatype) for choice in list(cat_param.choices))  # type: ignore
                for datatype in [int, float, str]
            )
            for cat_param in categoricals
        ):
            # If both checks pass, convert the space to a classic SearchSpace
            classic_space: dict[str, Any] = {}
            for key, value in space.get_attrs().items():
                if isinstance(value, Categorical):
                    classic_space[key] = neps.HPOCategorical(
                        choices=list(set(value.choices)),  # type: ignore
                        prior=value.choices[value.prior] if value.has_prior else None,  # type: ignore
                        prior_confidence=(
                            value.prior_confidence.value if value.has_prior else "low"
                        ),
                    )
                elif isinstance(value, Integer):
                    classic_space[key] = neps.HPOInteger(
                        lower=value.min_value,
                        upper=value.max_value,
                        prior=value.prior if value.has_prior else None,
                        prior_confidence=(
                            value.prior_confidence.value if value.has_prior else "low"
                        ),
                    )
                elif isinstance(value, Float):
                    classic_space[key] = neps.HPOFloat(
                        lower=value.min_value,
                        upper=value.max_value,
                        prior=value.prior if value.has_prior else None,
                        prior_confidence=(
                            value.prior_confidence.value if value.has_prior else "low"
                        ),
                    )
                elif isinstance(value, Fidelity):
                    if isinstance(value._domain, Integer):
                        classic_space[key] = neps.HPOInteger(
                            lower=value._domain.min_value,
                            upper=value._domain.max_value,
                            is_fidelity=True,
                        )
                    elif isinstance(value._domain, Float):
                        classic_space[key] = neps.HPOFloat(
                            lower=value._domain.min_value,
                            upper=value._domain.max_value,
                            is_fidelity=True,
                        )
                else:
                    classic_space[key] = neps.HPOConstant(value)
            return convert_mapping(classic_space)
    return None


def check_neps_space_compatibility(
    optimizer_to_check: (
        algorithms.OptimizerChoice
        | Mapping[str, Any]
        | tuple[algorithms.OptimizerChoice, Mapping[str, Any]]
        | Callable[
            Concatenate[SearchSpace, ...], optimizer.AskFunction
        ]  # Hack, while we transit
        | Callable[
            Concatenate[PipelineSpace, ...], optimizer.AskFunction
        ]  # from SearchSpace to
        | Callable[
            Concatenate[SearchSpace | PipelineSpace, ...], optimizer.AskFunction
        ]  # Pipeline
        | algorithms.CustomOptimizer
        | Literal["auto"]
    ) = "auto",
) -> Literal["neps", "classic", "both"]:
    """Check if the given optimizer is compatible with a NePS space.
    This function checks if the optimizer is a NePS-specific algorithm,
    a classic algorithm, or a combination of both.

    Args:
        optimizer_to_check: The optimizer to check for compatibility.
            It can be a NePS-specific algorithm, a classic algorithm,
            or a combination of both.

    Returns:
        A string indicating the compatibility:
            - "neps" if the optimizer is a NePS-specific algorithm,
            - "classic" if the optimizer is a classic algorithm,
            - "both" if the optimizer is a combination of both.
    """
    inner_optimizer = None
    if isinstance(optimizer_to_check, partial):
        inner_optimizer = optimizer_to_check.func
        while isinstance(inner_optimizer, partial):
            inner_optimizer = inner_optimizer.func

    only_neps_algorithm = (
        optimizer_to_check
        in (
            algorithms.neps_random_search,
            algorithms.neps_priorband,
            algorithms.complex_random_search,
        )
        or (
            inner_optimizer
            and inner_optimizer
            in (
                algorithms.neps_random_search,
                algorithms.neps_priorband,
                algorithms.complex_random_search,
            )
        )
        or optimizer_to_check == "auto"
        or (
            optimizer_to_check[0]
            in (
                "neps_random_search",
                "neps_priorband",
                "complex_random_search",
            )
            if isinstance(optimizer_to_check, tuple)
            else False
        )
        or (
            optimizer_to_check
            in (
                "neps_random_search",
                "neps_priorband",
                "complex_random_search",
            )
            if isinstance(optimizer_to_check, str)
            else False
        )
    )
    if only_neps_algorithm:
        return "neps"
    neps_and_classic_algorithm = (
        optimizer_to_check
        in (
            algorithms.random_search,
            algorithms.priorband,
        )
        or (
            inner_optimizer
            and inner_optimizer
            in (
                algorithms.random_search,
                algorithms.priorband,
            )
        )
        or optimizer_to_check == "auto"
        or (
            optimizer_to_check[0]
            in (
                "random_search",
                "priorband",
            )
            if isinstance(optimizer_to_check, tuple)
            else False
        )
        or (
            optimizer_to_check
            in (
                "random_search",
                "priorband",
            )
            if isinstance(optimizer_to_check, str)
            else False
        )
    )
    if neps_and_classic_algorithm:
        return "both"
    return "classic"
