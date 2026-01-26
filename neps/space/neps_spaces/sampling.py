"""This module defines various samplers for NEPS spaces, allowing for different sampling
strategies such as predefined values, random sampling, and mutation-based sampling.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
from scipy import stats

from neps.sampling.priors import PRIOR_CONFIDENCE_MAPPING
from neps.space.neps_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Domain,
    Float,
    Integer,
    PipelineSpace,
)
from neps.validation import validate_parameter_value

T = TypeVar("T")
P = TypeVar("P", bound="PipelineSpace")


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

        Args:
            domain_obj: The domain object to sample from.
            current_path: The current path in the resolution context.

        Returns:
            A sampled value of type T from the domain.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError()


class OnlyPredefinedValuesSampler(DomainSampler):
    """A sampler that only returns predefined values for a given path.
    If the path is not found in the predefined values, it raises a ValueError.

    Args:
        predefined_samplings: A mapping of paths to predefined values.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        """Initialize the sampler with predefined samplings.

        Args:
            predefined_samplings: A mapping of paths to predefined values.

        Raises:
            ValueError: If predefined_samplings is empty.
        """
        self._predefined_samplings = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],  # noqa: ARG002
        current_path: str,
    ) -> T:
        """Sample a value from the predefined samplings for the given path.

        Args:
            domain_obj: The domain object, not used in this sampler.
            current_path: The path for which to sample a value.

        Returns:
            The predefined value for the given path.

        Raises:
            ValueError: If the current path is not in the predefined samplings.
        """
        if current_path not in self._predefined_samplings:
            raise ValueError(f"No predefined value for path: {current_path!r}.")
        return cast("T", self._predefined_samplings[current_path])


class RandomSampler(DomainSampler):
    """A sampler that randomly samples from a predefined set of values.
    If the current path is not in the predefined values, it samples from the domain.

    Args:
        predefined_samplings: A mapping of paths to predefined values.
            This sampler will use these values if available, otherwise it will sample
            from the domain.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        """Initialize the sampler with predefined samplings.

        Args:
            predefined_samplings: A mapping of paths to predefined values.

        Raises:
            ValueError: If predefined_samplings is empty.
        """
        self._predefined_samplings = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Sample a value from the predefined samplings or the domain.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value.

        Returns:
            A sampled value, either from the predefined samplings or from the
                domain.

        Raises:
            ValueError: If the current path is not in the predefined samplings and
                the domain does not have a prior defined.
        """
        if current_path not in self._predefined_samplings:
            sampled_value = domain_obj.sample()
        else:
            sampled_value = cast("T", self._predefined_samplings[current_path])
        return sampled_value


class IOSampler(DomainSampler):
    """A sampler that samples by asking the user at each decision."""

    def __call__(  # noqa: C901, PLR0912
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Sample a value from the predefined samplings or the domain.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The current path in the search space.

        Returns:
            A value from the user input.
        """
        if isinstance(domain_obj, Float | Integer):
            print(
                "Please provide"
                f" {'a float' if isinstance(domain_obj, Float) else 'an integer'} value"
                f" for \n\t'{current_path}'\nin the range [{domain_obj.lower},"  # type: ignore[attr-defined]
                f" {domain_obj.upper}]: ",  # type: ignore[attr-defined]
            )
        elif isinstance(domain_obj, Categorical):
            from neps.space.neps_spaces.string_formatter import format_value

            # Format choices and check for multi-line content
            formatted_choices = [format_value(c, indent=0) for c in domain_obj.choices]  # type: ignore[attr-defined, arg-type]
            has_multiline = any("\n" in formatted for formatted in formatted_choices)

            # Build choices display
            choices_lines = [""] if has_multiline else []
            for n, formatted in enumerate(formatted_choices):
                if "\n" in formatted:
                    choices_lines.append(f"Option {n}:")
                    choices_lines.append(formatted)
                else:
                    choices_lines.append(f"Option {n}: {formatted}")
                if has_multiline and n < len(formatted_choices) - 1:
                    choices_lines.append("")  # Blank line separator between options

            choices_list = "\n".join(choices_lines)
            max_index = int(domain_obj.range_compatibility_identifier) - 1  # type: ignore[attr-defined]
            print(
                f"Please provide an index for '{current_path}'\n"
                f"Choices:\n{choices_list}\n"
                f"Valid range: [0, {max_index}]: "
            )

        while True:
            sampled_value: str | int | float = input()
            try:
                if isinstance(domain_obj, Integer):
                    sampled_value = int(sampled_value)
                elif isinstance(domain_obj, Float):
                    sampled_value = float(sampled_value)
                elif isinstance(domain_obj, Categorical):
                    sampled_value = int(sampled_value)
                else:
                    raise ValueError(
                        f"Unsupported domain type: {type(domain_obj).__name__}"
                    )

                assert isinstance(domain_obj, Float | Integer | Categorical)

                if validate_parameter_value(domain_obj, sampled_value):
                    print(f"Value {sampled_value} recorded.\n")
                    break
                else:
                    print(
                        f"Invalid value '{sampled_value}' for domain '{current_path}'. "
                        "Please try again: ",
                    )
            except ValueError:
                print(
                    f"Could not convert input '{sampled_value}' to the required type. "
                    "Please try again: ",
                )

        return cast("T", sampled_value)

    def sample_environment_values(self, pipeline_space: P) -> Mapping[str, Any]:
        """Get the environment values for the sampler.

        Returns:
            The interactively chosen environment values.
        """
        environment_values = {}
        for fidelity_name, fidelity_object in pipeline_space.fidelity_attrs.items():
            domain_obj = fidelity_object.domain
            print(
                "Please provide"
                f" {'a float' if isinstance(domain_obj, Float) else 'an integer'} value"
                f" for the Fidelity '{fidelity_name}' in the range"
                f" [{domain_obj.lower}, {domain_obj.upper}]: ",
            )
            while True:
                sampled_value: str | int | float = input()
                try:
                    if isinstance(domain_obj, Integer):
                        sampled_value = int(sampled_value)
                    elif isinstance(domain_obj, Float):
                        sampled_value = float(sampled_value)
                    else:
                        raise ValueError(
                            f"Unsupported domain type: {type(domain_obj).__name__}"
                        )

                    if validate_parameter_value(domain_obj, sampled_value):
                        print(f"Value {sampled_value} recorded.\n")
                        break
                    else:
                        print(
                            f"Invalid value '{sampled_value}' for Fidelity"
                            f" '{fidelity_object!s}'. Please try again: ",
                        )
                except ValueError:
                    print(
                        f"Could not convert input '{sampled_value}' to the required type."
                        " Please try again: ",
                    )
            environment_values[fidelity_name] = sampled_value

        return environment_values


class PriorOrFallbackSampler(DomainSampler):
    """A sampler that uses a prior value if available, otherwise falls back to another
    sampler.

    Args:
        fallback_sampler: A DomainSampler to use if the prior is not available.
        always_use_prior: If True, always use the prior value when available.
    """

    def __init__(
        self,
        fallback_sampler: DomainSampler,
        always_use_prior: bool = False,  # noqa: FBT001, FBT002
    ):
        """Initialize the sampler with a fallback sampler and a flag to always use the
        prior.

        Args:
            fallback_sampler: A DomainSampler to use if the prior is not available.
            always_use_prior: If True, always use the prior value when available.
        """
        self._fallback_sampler = fallback_sampler
        self._always_use_prior = always_use_prior

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Sample a value from the domain, using the prior if available and according to
        the prior confidence probability.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value.

        Returns:
            A sampled value, either from the prior or from the fallback sampler.

        Raises:
            ValueError: If the domain does not have a prior defined and the fallback
                sampler is not provided.
        """
        if domain_obj.has_prior:
            _prior_probability = PRIOR_CONFIDENCE_MAPPING.get(
                domain_obj.prior_confidence.value, 0.5
            )
            if isinstance(domain_obj, Categorical) or self._always_use_prior:
                if (
                    random.choices(
                        (True, False),
                        weights=(_prior_probability, 1 - _prior_probability),
                        k=1,
                    )[0]
                    or self._always_use_prior
                ):
                    # If the prior is defined, we sample from it.
                    return domain_obj.prior

            # For Integers and Floats, sample gaussians around the prior

            elif isinstance(domain_obj, Integer | Float):
                # Sample an integer from a Gaussian distribution centered around the
                # prior, cut of the tails to ensure the value is within the domain's
                # range. Using the _prior_probability to determine the standard deviation
                assert hasattr(domain_obj, "lower")
                assert hasattr(domain_obj, "upper")
                assert hasattr(domain_obj, "prior")

                std_dev = 1 / (
                    10 * _prior_probability / (domain_obj.upper - domain_obj.lower)  # type: ignore
                )

                a = (domain_obj.lower - domain_obj.prior) / std_dev  # type: ignore
                b = (domain_obj.upper - domain_obj.prior) / std_dev  # type: ignore
                sampled_value = stats.truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=domain_obj.prior,  # type: ignore
                    scale=std_dev,
                )
                if isinstance(domain_obj, Integer):
                    sampled_value = round(sampled_value)
                else:
                    sampled_value = float(sampled_value)  # type: ignore
                return cast("T", sampled_value)

        return self._fallback_sampler(
            domain_obj=domain_obj,
            current_path=current_path,
        )


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


class MutateByForgettingSampler(DomainSampler):
    """A sampler that mutates predefined samplings by forgetting a certain number of
    them. It randomly selects a number of predefined samplings to forget and returns a
    new sampler that only uses the remaining samplings.

    Args:
        predefined_samplings: A mapping of paths to predefined values.
        n_forgets: The number of predefined samplings to forget.
            This should be an integer greater than 0 and less than or equal to the
            number of predefined samplings.

    Raises:
        ValueError: If n_forgets is not a valid integer or if it exceeds the number
            of predefined samplings.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_forgets: int,
    ):
        """Initialize the sampler with predefined samplings and a number of forgets.

        Args:
            predefined_samplings: A mapping of paths to predefined values.
            n_forgets: The number of predefined samplings to forget.
                This should be an integer greater than 0 and less than or equal to the
                number of predefined samplings.

        Raises:
            ValueError: If n_forgets is not a valid integer or if it exceeds the
                number of predefined samplings.
        """
        if (
            not isinstance(n_forgets, int)
            or n_forgets <= 0
            or n_forgets > len(predefined_samplings)
        ):
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
        """Sample a value from the mutated predefined samplings or the domain.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value.

        Returns:
            A sampled value, either from the mutated predefined samplings or from
                the domain.

        Raises:
            ValueError: If the current path is not in the mutated predefined
                samplings and the domain does not have a prior defined.
        """
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)


class MutatateUsingCentersSampler(DomainSampler):
    """A sampler that mutates predefined samplings by forgetting a certain number of them,
    but still uses the original values as centers for sampling.

    Args:
        predefined_samplings: A mapping of paths to predefined values.
        n_mutations: The number of predefined samplings to mutate.
            This should be an integer greater than 0 and less than or equal to the number
            of predefined samplings.

    Raises:
        ValueError: If n_mutations is not a valid integer or if it exceeds the number
            of predefined samplings.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_mutations: int,
    ):
        """Initialize the sampler with predefined samplings and a number of mutations.

        Args:
            predefined_samplings: A mapping of paths to predefined values.
            n_mutations: The number of predefined samplings to mutate.
                This should be an integer greater than 0 and less than or equal to the
                number of predefined samplings.

        Raises:
            ValueError: If n_mutations is not a valid integer or if it exceeds
                the number of predefined samplings.
        """
        if (
            not isinstance(n_mutations, int)
            or n_mutations <= 0
            or n_mutations > len(predefined_samplings)
        ):
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
        """Sample a value from the predefined samplings or the domain, using original
        values as centers if the current path is not in the kept samplings.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value.

        Returns:
            A sampled value, either from the kept samplings or from the domain,
                using the original values as centers if necessary.

        Raises:
            ValueError: If the current path is not in the kept samplings and the
                domain does not have a prior defined.
        """
        if current_path not in self._kept_samplings_to_make:
            # For this path we either have forgotten the value or we never had it.
            if current_path in self._original_samplings_to_make:
                # We had a value for this path originally, use it as a center.
                original_value = self._original_samplings_to_make[current_path]
                sampled_value = domain_obj.centered_around(
                    center=original_value,
                    confidence=ConfidenceLevel.MEDIUM,
                ).sample()
            else:
                # We never had a value for this path, we can only sample from the domain.
                sampled_value = domain_obj.sample()
        else:
            # For this path we have chosen to keep the original value.
            sampled_value = cast("T", self._kept_samplings_to_make[current_path])

        return sampled_value


class CrossoverNotPossibleError(Exception):
    """Exception raised when a crossover operation is not possible."""


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


class CrossoverByMixingSampler(DomainSampler):
    """A sampler that performs a crossover operation by mixing two sets of predefined
    samplings. It combines the predefined samplings from two sources, allowing for a
    probability-based selection of values from either source.

    Args:
        predefined_samplings_1: The first set of predefined samplings.
        predefined_samplings_2: The second set of predefined samplings.
        prefer_first_probability: The probability of preferring values from the first
            set over the second set when both have values for the same path.
            This should be a float between 0 and 1, where 0 means always prefer the
            second set and 1 means always prefer the first set.

    Raises:
        ValueError: If prefer_first_probability is not between 0 and 1.
        CrossoverNotPossibleError: If no crossovers were made between the two sets
            of predefined samplings.
    """

    def __init__(
        self,
        predefined_samplings_1: Mapping[str, Any],
        predefined_samplings_2: Mapping[str, Any],
        prefer_first_probability: float,
    ):
        """Initialize the sampler with two sets of predefined samplings and a preference
        probability for the first set.

        Args:
            predefined_samplings_1: The first set of predefined samplings.
            predefined_samplings_2: The second set of predefined samplings.
            prefer_first_probability: The probability of preferring values from the
                first set over the second set when both have values for the same path.
                This should be a float between 0 and 1, where 0 means always prefer the
                second set and 1 means always prefer the first set.

        Raises:
            ValueError: If prefer_first_probability is not between 0 and 1.
        """
        if not isinstance(prefer_first_probability, float) or not (
            0 <= prefer_first_probability <= 1
        ):
            raise ValueError(
                "Invalid value for `prefer_first_probability`:"
                f" {prefer_first_probability!r}."
            )

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
        """Sample a value from the crossed-over predefined samplings or the domain.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value.

        Returns:
            A sampled value, either from the crossed-over predefined samplings or
                from the domain.

        Raises:
            ValueError: If the current path is not in the crossed-over predefined
                samplings and the domain does not have a prior defined.
        """
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)


class GridSampler(DomainSampler):
    """A sampler that records when sampling a value (or reusing an already sampled value)."""

    def __init__(
        self,
        sampling_density: int = 5,
    ):
        """Initialize the sampler with predefined samplings.

        Args:
            predefined_samplings: A mapping of paths to predefined values.
        """
        self._sampling_density = sampling_density

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Samples a value.

        Args:
            domain_obj: The domain object, not used in this sampler.
            current_path: The path for which to sample a value.

        Returns:
            The sampled value for the given path.
        """
        if isinstance(domain_obj, Float | Integer):
            sampling_ratio = random.randint(0, self._sampling_density - 1) / (
                self._sampling_density - 1
            )
            if domain_obj.log:
                lower_log = np.log(domain_obj.lower)
                upper_log = np.log(domain_obj.upper)
                sampled_value = float(
                    np.exp(lower_log + (upper_log - lower_log) * sampling_ratio)
                )
                if isinstance(domain_obj, Integer):
                    sampled_value = round(sampled_value)
            else:
                sampled_value = (
                    domain_obj.lower
                    + (domain_obj.upper - domain_obj.lower) * sampling_ratio
                )
                if isinstance(domain_obj, Integer):
                    sampled_value = round(sampled_value)
        elif isinstance(domain_obj, Categorical):
            sampled_value = random.randint(0, len(domain_obj.choices) - 1)  # type: ignore[attr-defined]

        else:
            raise ValueError(f"Unsupported domain type: {type(domain_obj).__name__}")

        return cast("T", sampled_value)


class QueueBasedSampler(DomainSampler):
    """A sampler that returns values based on a queue of indices for exhaustive grid search.

    This sampler is designed for dynamic discovery of the search space structure.
    It maintains a queue of tuples (param_type, max_value, current_index) and returns
    values based on the current_index for each parameter.

    The queue is managed externally and grows dynamically as new parameters are discovered
    during sampling. When a parameter is first encountered, it's appended to the queue
    with current_index=0.

    Args:
        queue: A reference to the queue list. Modified in-place when new parameters
            are discovered.
        sampling_density: The number of grid points to use for numerical parameters
            (Float and Integer).
    """

    def __init__(
        self,
        queue: list[tuple[str, int, int]],
        sampling_density: int = 5,
    ):
        """Initialize the sampler with a queue reference and sampling density.

        Args:
            queue: A reference to the queue list to use for tracking sampling decisions.
            sampling_density: The number of grid points for numerical parameters.
        """
        self._queue = queue
        self._sampling_density = sampling_density
        self._queue_position = 0  # Track where we are in queue during sampling

    def reset_position(self):
        """Reset the queue position to the start for the next configuration sampling."""
        self._queue_position = 0

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,  # noqa: ARG002
    ) -> T:
        """Sample a value based on the next queue entry.

        If this is a newly discovered parameter (queue_position >= queue length),
        append a new entry to the queue with current_index=0.

        Args:
            domain_obj: The domain object from which to sample.
            current_path: The path for which to sample a value (not used).

        Returns:
            A sampled value based on the queue entry's current_index.

        Raises:
            ValueError: If the domain type is unsupported or if there's a type mismatch.
        """
        # Handle degenerate numerical ranges first (they are never added to queue)
        if (
            isinstance(domain_obj, Float | Integer)
            and domain_obj.lower == domain_obj.upper
        ):  # type: ignore[attr-defined]
            # Degenerate range: just return the single value without using queue
            value = domain_obj.lower  # type: ignore[attr-defined]
            if isinstance(domain_obj, Integer):
                value = round(float(value))
            else:
                value = float(value)
            return cast("T", value)

        # Check if this is a newly discovered parameter
        if self._queue_position >= len(self._queue):
            # Append new entry to queue with index 0 (not for degenerate ranges, handled above)
            if isinstance(domain_obj, Categorical):
                n_choices = len(domain_obj.choices)  # type: ignore[attr-defined]
                self._queue.append(("cat", n_choices, 0))
            elif isinstance(domain_obj, Float | Integer):
                # Non-degenerate numerical (degenerate already handled above)
                self._queue.append(("num", self._sampling_density, 0))
            else:
                raise ValueError(f"Unsupported domain type: {type(domain_obj).__name__}")

        # Get the current queue entry for non-degenerate parameters
        param_type, max_value, current_index = self._queue[self._queue_position]
        self._queue_position += 1

        # Return appropriate value based on domain type
        if isinstance(domain_obj, Categorical):
            if param_type != "cat":
                raise ValueError(
                    f"Queue type mismatch: expected 'cat', got '{param_type}'"
                )
            # Return the choice index directly
            return cast("T", current_index)

        if isinstance(domain_obj, Float | Integer):
            if param_type != "num":
                raise ValueError(
                    f"Queue type mismatch: expected 'num', got '{param_type}'"
                )

            # Generate grid values and return the one at current_index
            if domain_obj.log:
                lower_log = np.log(domain_obj.lower)  # type: ignore[attr-defined]
                upper_log = np.log(domain_obj.upper)  # type: ignore[attr-defined]
                values = np.logspace(lower_log, upper_log, max_value, base=np.e)
            else:
                values = np.linspace(
                    domain_obj.lower,  # type: ignore[attr-defined]
                    domain_obj.upper,  # type: ignore[attr-defined]
                    max_value,
                )

            value = values[current_index]
            if isinstance(domain_obj, Integer):
                value = round(float(value))
            else:
                value = float(value)

            return cast("T", value)

        raise ValueError(f"Unsupported domain type: {type(domain_obj).__name__}")
