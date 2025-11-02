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
    _UNSET,
    Categorical,
    Domain,
    Fidelity,
    Float,
    Integer,
    Lazy,
    Operation,
    PipelineSpace,
    Repeated,
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


def construct_sampling_path(
    path_parts: list[str],
    domain_obj: Domain,
) -> str:
    """Construct a sampling path for a domain object.

    The sampling path uniquely identifies a sampled value in the resolution context.
    It consists of the hierarchical path through the pipeline space and a domain
    identifier that includes type and range information.

    Args:
        path_parts: The hierarchical path parts (e.g., ["Resolvable", "integer1"]).
        domain_obj: The domain object for which to construct the path.

    Returns:
        A string representing the full sampling path in the format:
        "<path.parts>::<type>__<range_compatibility_identifier>"
        Example: "Resolvable.integer1::integer__0_1_False"

    Raises:
        ValueError: If path_parts is empty or domain_obj is not a Domain.
    """
    if not path_parts:
        raise ValueError("path_parts cannot be empty")
    if not isinstance(domain_obj, Domain):
        raise ValueError(f"domain_obj must be a Domain, got {type(domain_obj)}")

    # Get the domain type name (e.g., "integer", "float", "categorical")
    domain_obj_type_name = type(domain_obj).__name__.lower()

    # Get the range compatibility identifier (e.g., "0_1_False" for
    # Integer(0, 1, log=False))
    range_compatibility_identifier = domain_obj.range_compatibility_identifier

    # Combine type and range: "integer__0_1_False"
    domain_obj_identifier = f"{domain_obj_type_name}__{range_compatibility_identifier}"

    # Join path parts with dots: "Resolvable.integer1"
    current_path = ".".join(path_parts)

    # Append domain identifier: "Resolvable.integer1::integer__0_1_False"
    current_path += "::" + domain_obj_identifier

    return current_path


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

        # Construct the unique sampling path for this domain object
        current_path = construct_sampling_path(
            path_parts=self._current_path_parts,
            domain_obj=domain_obj,
        )

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
                # We need special handling if we are dealing with a "choice provider",
                # which will select a tuple of choices from its own choices,
                # from which then this original categorical will pick.

                # Ideally, from the choices provided, we want to first pick one,
                # and then only resolve that picked item.
                # We don't want the resolution process to directly go inside
                # the tuple of provided choices that gets picked from the provider,
                # since that would lead to potentially exponential growth
                # and in resolving stuff that will ultimately be useless to us.

                # For this reason, if we haven't already sampled this categorical
                # (the choice provider), we make sure to wrap each of the choices
                # inside it in a lazy resolvable.
                # This ensures that the resolving process stops directly after
                # the provider has made its choice.

                # Since we may be manually creating a new categorical object
                # for the provider, which is what will then get resolved,
                # it's important that we manually store
                # in the context that resolved value for the original object.
                # The original object can possibly be reused elsewhere.

                if isinstance(
                    initial_attr_value, Categorical
                ) and context.was_already_resolved(initial_attr_value):
                    # Before making adjustments, we make sure we haven't
                    # already chosen a value for the provider.
                    # Otherwise, we already have the final answer for it.
                    resolved_attr_value = context.get_resolved(initial_attr_value)
                elif isinstance(initial_attr_value, Categorical) or (
                    isinstance(initial_attr_value, Resampled)
                    and isinstance(initial_attr_value.source, Categorical)
                ):
                    # We have a previously unseen provider.
                    # Create a new object where the choices are lazy,
                    # and then sample from it, manually tracking the context.

                    choice_provider_final_attrs = {**initial_attr_value.get_attrs()}
                    choice_provider_choices = choice_provider_final_attrs["choices"]
                    if isinstance(choice_provider_choices, tuple | list):
                        choice_provider_choices = tuple(
                            Lazy(content=choice) for choice in choice_provider_choices
                        )
                    choice_provider_final_attrs["choices"] = choice_provider_choices
                    choice_provider_adjusted = initial_attr_value.from_attrs(
                        choice_provider_final_attrs
                    )

                    resolved_attr_value = self._resolve(
                        choice_provider_adjusted, "choice_provider", context
                    )
                    if not isinstance(initial_attr_value, Resampled):
                        # It's important that we handle filling the context here,
                        # as we manually created a different object from the original.
                        # In case the original categorical is used again,
                        # it will need to be reused with the final value we resolved.
                        context.add_resolved(initial_attr_value, resolved_attr_value)
                else:
                    # We have "choices" which are ready to use.
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

        initial_attrs = operation_obj.get_attrs()
        final_attrs = {}
        needed_resolving = False

        for attr_name, initial_attr_value in initial_attrs.items():
            resolved_attr_value = self._resolve(initial_attr_value, attr_name, context)

            # Special handling for 'args': if it was a Resolvable that resolved to a
            # non-iterable, wrap it in a tuple since Operation expects args to be a
            # sequence
            if (
                attr_name == "args"
                and isinstance(initial_attr_value, Resolvable)
                and not isinstance(resolved_attr_value, tuple | list | Resolvable)
            ):
                resolved_attr_value = (resolved_attr_value,)

            final_attrs[attr_name] = resolved_attr_value
            needed_resolving = needed_resolving or (
                initial_attr_value is not resolved_attr_value
            )

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

        if resolvable_to_resample_obj is resampled_obj.source:
            # The final resolvable we are resolving needs to be a different
            # instance from the original wrapped object.
            # Otherwise, it's possible we'll be taking its result
            # from the context cache, instead of resampling it.
            raise ValueError(
                "The final object must be a different instance from the original: "
                f"{resolvable_to_resample_obj!r}"
            )

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

        if not fidelity_obj.lower <= result <= fidelity_obj.upper:
            raise ValueError(
                f"Value for fidelity with name {fidelity_name!r} is outside its allowed"
                " range "
                + f"[{fidelity_obj.lower!r}, {fidelity_obj.upper!r}]. "
                + f"Received: {result!r}."
            )

        context.add_resolved(fidelity_obj, result)
        return result

    @_resolver_dispatch.register
    def _(
        self,
        repeated_resolvable_obj: Repeated,
        context: SamplingResolutionContext,
    ) -> tuple[Any]:
        if context.was_already_resolved(repeated_resolvable_obj):
            return context.get_resolved(repeated_resolvable_obj)

        # First figure out how many times we need to resolvable repeated,
        # then do that many resolves of that object.
        # It does not matter what type the content is.
        # Return all the results as a tuple.

        unresolved_count = repeated_resolvable_obj.count
        resolved_count = self._resolve(unresolved_count, "repeat_count", context)

        if not isinstance(resolved_count, int):
            raise ValueError(
                f"The resolved count value for {repeated_resolvable_obj!r} is not an int."
                f" Resolved to {resolved_count!r}"
            )

        obj_to_repeat = repeated_resolvable_obj.content
        result = []
        for i in range(resolved_count):
            result.append(self._resolve(obj_to_repeat, f"repeated_item[{i}]", context))
        result = tuple(result)  # type: ignore[assignment]

        context.add_resolved(repeated_resolvable_obj, result)
        return result  # type: ignore[return-value]

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: Lazy,
        context: SamplingResolutionContext,  # noqa: ARG002
    ) -> Any:
        # When resolving a lazy resolvable,
        # just directly return the content it's holding.
        # The purpose of the lazy resolvable is to stop
        # the resolver from going deeper into the process.
        # In this case, to stop the resolution of `resolvable_obj.content`.
        # No need to add it in the resolved cache.
        return resolvable_obj.content

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: dict,
        context: SamplingResolutionContext,
    ) -> dict[Any, Any]:
        # The logic below is done so that if the original dict
        # had only things that didn't need resolving,
        # we return the original object.
        # That is important for the rest of the resolving process.
        original_dict = resolvable_obj
        new_dict = {}
        needed_resolving = False

        for k, initial_v in original_dict.items():
            resolved_v = self._resolve(initial_v, f"mapping_value{{{k}}}", context)
            new_dict[k] = resolved_v
            needed_resolving = needed_resolving or (resolved_v is not initial_v)

        result = original_dict
        if needed_resolving:
            result = new_dict

        # TODO: [lum] reconsider this below. We likely should cache them,
        #  similarly to other things.
        # IMPORTANT: Dicts are not stored in the resolved cache.
        # Otherwise, we won't go inside them the next time
        # and will ignore any resampled things inside.
        return result

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: tuple,
        context: SamplingResolutionContext,
    ) -> tuple[Any]:
        return self._resolve_sequence(resolvable_obj, context)  # type: ignore[return-value]

    @_resolver_dispatch.register
    def _(
        self,
        resolvable_obj: list,
        context: SamplingResolutionContext,
    ) -> list[Any]:
        return self._resolve_sequence(resolvable_obj, context)  # type: ignore[return-value]

    def _resolve_sequence(
        self,
        resolvable_obj: tuple | list,
        context: SamplingResolutionContext,
    ) -> tuple[Any] | list[Any]:
        # The logic below is done so that if the original sequence
        # had only things that didn't need resolving,
        # we return the original object.
        # That is important for the rest of the resolving process.
        original_sequence = resolvable_obj
        new_list = []
        needed_resolving = False

        for idx, initial_item in enumerate(original_sequence):
            resolved_item = self._resolve(initial_item, f"sequence[{idx}]", context)
            new_list.append(resolved_item)
            needed_resolving = needed_resolving or (initial_item is not resolved_item)

        result = original_sequence
        if needed_resolving:
            # We also want to return a result of the same type
            # as the original received sequence.
            original_type = type(original_sequence)
            result = original_type(new_list)

        # TODO: [lum] reconsider this below. We likely should cache them,
        #  similarly to other things.
        # IMPORTANT: Sequences are not stored in the resolved cache.
        # Otherwise, we won't go inside them the next time
        # and will ignore any resampled things inside.
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

    operation_args: list[Any] = []
    for arg in operation.args:
        if isinstance(arg, tuple | list):
            arg_sequence: list[Any] = []
            for a in arg:
                converted_arg = (
                    convert_operation_to_callable(a) if isinstance(a, Operation) else a
                )
                arg_sequence.append(converted_arg)
            if isinstance(arg, tuple):
                operation_args.append(tuple(arg_sequence))
            else:
                operation_args.append(arg_sequence)
        else:
            operation_args.append(
                convert_operation_to_callable(arg) if isinstance(arg, Operation) else arg
            )

    operation_kwargs: dict[str, Any] = {}
    for kwarg_name, kwarg_value in operation.kwargs.items():
        if isinstance(kwarg_value, tuple | list):
            kwarg_sequence: list[Any] = []
            for a in kwarg_value:
                converted_kwarg = (
                    convert_operation_to_callable(a) if isinstance(a, Operation) else a
                )
                kwarg_sequence.append(converted_kwarg)
            if isinstance(kwarg_value, tuple):
                operation_kwargs[kwarg_name] = tuple(kwarg_sequence)
            else:
                operation_kwargs[kwarg_name] = kwarg_sequence
        else:
            operation_kwargs[kwarg_name] = (
                convert_operation_to_callable(kwarg_value)
                if isinstance(kwarg_value, Operation)
                else kwarg_value
            )

    return cast(Callable, operator(*operation_args, **operation_kwargs))


def _serialize_operation(operation: Operation | str | Callable) -> str:
    """Serialize an operation to its string representation.

    This is a helper function to convert Operation objects to strings
    for inclusion in the operands field of UnwrappedConfigStringPart.
    """
    if isinstance(operation, str):
        return operation

    # Handle non-Operation objects (e.g., resolved PyTorch modules, integers, etc.)
    if not isinstance(operation, Operation):
        return str(operation)

    # For Operation objects, build the string representation
    operator_name = (
        operation.operator
        if isinstance(operation.operator, str)
        else operation.operator.__name__
    )

    if not operation.args:
        # No operands - just return operator name
        return operator_name

    # Recursively serialize operands
    operand_strs = [_serialize_operation(arg) for arg in operation.args]
    return f"{operator_name}({', '.join(operand_strs)})"


def _operation_to_unwrapped_config(
    operation: Operation | str,
    level: int = 1,
    opening_index: int = 0,
) -> tuple[list[config_string.UnwrappedConfigStringPart], int]:
    """Convert an Operation to unwrapped config parts.

    Returns:
        A tuple of (list of parts, next available opening_index)
    """
    result = []

    if isinstance(operation, Operation):
        operator = operation.operator
        kwargs = str(operation.kwargs)

        # Build operands string and collect child parts
        operand_strs = []
        all_child_parts = []
        next_opening = opening_index + 1

        for operand in operation.args:
            if isinstance(operand, Operation):
                # Only create child parts if the operation has operands
                # (otherwise it's just a simple name like "ReLU")
                if operand.args:
                    # Recursively get unwrapped parts for the nested operation
                    child_parts, next_opening = _operation_to_unwrapped_config(
                        operand, level + 1, next_opening
                    )
                    all_child_parts.extend(child_parts)
                # Serialize this operand to a string for the operands field
                operand_strs.append(_serialize_operation(operand))
            else:
                operand_strs.append(str(operand))

        # Create operands string
        operands_str = ", ".join(operand_strs)

        item = config_string.UnwrappedConfigStringPart(
            level=level,
            opening_index=opening_index,
            operator=operator,
            hyperparameters=kwargs,
            operands=operands_str,
        )
        result.append(item)
        result.extend(all_child_parts)

        return result, next_opening
    item = config_string.UnwrappedConfigStringPart(
        level=level,
        opening_index=opening_index,
        operator=operation,
        hyperparameters="",
        operands="",
    )
    return [item], opening_index + 1


def convert_operation_to_string(operation: Operation | str | int | float) -> str:
    """Convert an Operation to a string representation.

    Args:
        operation: The Operation to convert, or a primitive value.

    Returns:
        A string representation of the operation or value.

    Raises:
        ValueError: If the operation is not a valid Operation object.
    """
    # Handle non-Operation values (resolved primitives)
    if not isinstance(operation, Operation):
        return str(operation)

    unwrapped_config, _ = _operation_to_unwrapped_config(operation)
    return config_string.wrap_config_into_string(tuple(unwrapped_config))


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
                        lower=value.lower,
                        upper=value.upper,
                        log=value._log if hasattr(value, "_log") else False,
                        prior=value.prior if value.has_prior else None,
                        prior_confidence=(
                            value.prior_confidence.value if value.has_prior else "low"
                        ),
                    )
                elif isinstance(value, Float):
                    classic_space[key] = neps.HPOFloat(
                        lower=value.lower,
                        upper=value.upper,
                        log=value._log if hasattr(value, "_log") else False,
                        prior=value.prior if value.has_prior else None,
                        prior_confidence=(
                            value.prior_confidence.value if value.has_prior else "low"
                        ),
                    )
                elif isinstance(value, Fidelity):
                    if isinstance(value._domain, Integer):
                        classic_space[key] = neps.HPOInteger(
                            lower=value._domain.lower,
                            upper=value._domain.upper,
                            log=(
                                value._domain._log
                                if hasattr(value._domain, "_log")
                                else False
                            ),
                            is_fidelity=True,
                        )
                    elif isinstance(value._domain, Float):
                        classic_space[key] = neps.HPOFloat(
                            lower=value._domain.lower,
                            upper=value._domain.upper,
                            log=(
                                value._domain._log
                                if hasattr(value._domain, "_log")
                                else False
                            ),
                            is_fidelity=True,
                        )
                else:
                    classic_space[key] = neps.HPOConstant(value)
            return convert_mapping(classic_space)
    return None


def convert_classic_to_neps_search_space(
    space: SearchSpace,
) -> PipelineSpace:
    """Convert a classic SearchSpace to a NePS PipelineSpace if possible.
    This function converts a classic SearchSpace to a NePS PipelineSpace.

    Args:
        space: The classic SearchSpace to convert.

    Returns:
        A NePS PipelineSpace.
    """

    class NEPSSpace(PipelineSpace):
        """A NePS-specific PipelineSpace."""

    for parameter_name, parameter in space.elements.items():
        if isinstance(parameter, neps.HPOCategorical):
            setattr(
                NEPSSpace,
                parameter_name,
                Categorical(
                    choices=tuple(parameter.choices),
                    prior=(
                        parameter.choices.index(parameter.prior)
                        if parameter.prior
                        else _UNSET
                    ),
                    prior_confidence=(
                        parameter.prior_confidence
                        if parameter.prior_confidence
                        else _UNSET
                    ),
                ),
            )
        elif isinstance(parameter, neps.HPOConstant):
            setattr(NEPSSpace, parameter_name, parameter.value)
        elif isinstance(parameter, neps.HPOInteger):
            new_integer = Integer(
                lower=parameter.lower,
                upper=parameter.upper,
                log=parameter.log,
                prior=parameter.prior if parameter.prior else _UNSET,
                prior_confidence=(
                    parameter.prior_confidence if parameter.prior_confidence else _UNSET
                ),
            )
            setattr(
                NEPSSpace,
                parameter_name,
                (Fidelity(domain=new_integer) if parameter.is_fidelity else new_integer),
            )
        elif isinstance(parameter, neps.HPOFloat):
            new_float = Float(
                lower=parameter.lower,
                upper=parameter.upper,
                log=parameter.log,
                prior=parameter.prior if parameter.prior else _UNSET,
                prior_confidence=(
                    parameter.prior_confidence if parameter.prior_confidence else _UNSET
                ),
            )
            setattr(
                NEPSSpace,
                parameter_name,
                (Fidelity(domain=new_float) if parameter.is_fidelity else new_float),
            )

    return NEPSSpace()


ONLY_NEPS_ALGORITHMS_NAMES = [
    "neps_random_search",
    "neps_priorband",
    "complex_random_search",
    "neps_hyperband",
    "complex_hyperband",
]
CLASSIC_AND_NEPS_ALGORITHMS_NAMES = [
    "random_search",
    "priorband",
    "hyperband",
    "grid_search",
]


# Lazy initialization to avoid circular imports
def _get_only_neps_algorithms_functions() -> list[Callable]:
    """Get the list of NEPS-only algorithm functions lazily."""
    return [
        algorithms.neps_random_search,
        algorithms.neps_priorband,
        algorithms.complex_random_search,
        algorithms.neps_hyperband,
        algorithms.neps_grid_search,
    ]


def _get_classic_and_neps_algorithms_functions() -> list[Callable]:
    """Get the list of classic and NEPS algorithm functions lazily."""
    return [
        algorithms.random_search,
        algorithms.priorband,
        algorithms.hyperband,
        algorithms.grid_search,
    ]


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
        optimizer_to_check in _get_only_neps_algorithms_functions()
        or (inner_optimizer and inner_optimizer in _get_only_neps_algorithms_functions())
        or (
            optimizer_to_check[0] in ONLY_NEPS_ALGORITHMS_NAMES
            if isinstance(optimizer_to_check, tuple)
            else False
        )
        or (
            optimizer_to_check in ONLY_NEPS_ALGORITHMS_NAMES
            if isinstance(optimizer_to_check, str)
            else False
        )
    )
    if only_neps_algorithm:
        return "neps"
    neps_and_classic_algorithm = (
        optimizer_to_check in _get_classic_and_neps_algorithms_functions()
        or (
            inner_optimizer
            and inner_optimizer in _get_classic_and_neps_algorithms_functions()
        )
        or optimizer_to_check == "auto"
        or (
            optimizer_to_check[0] in CLASSIC_AND_NEPS_ALGORITHMS_NAMES
            if isinstance(optimizer_to_check, tuple)
            else False
        )
        or (
            optimizer_to_check in CLASSIC_AND_NEPS_ALGORITHMS_NAMES
            if isinstance(optimizer_to_check, str)
            else False
        )
    )
    if neps_and_classic_algorithm:
        return "both"
    return "classic"
