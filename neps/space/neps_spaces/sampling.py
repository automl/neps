"""This module defines various samplers for NEPS spaces, allowing for different sampling
strategies such as predefined values, random sampling, and mutation-based sampling.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

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
        return cast(T, self._predefined_samplings[current_path])


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
            sampled_value = cast(T, self._predefined_samplings[current_path])
        return sampled_value


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
                assert hasattr(domain_obj, "min_value")
                assert hasattr(domain_obj, "max_value")
                assert hasattr(domain_obj, "prior")

                std_dev = 1 / (
                    10
                    * _prior_probability
                    / (domain_obj.max_value - domain_obj.min_value)  # type: ignore
                )

                a = (domain_obj.min_value - domain_obj.prior) / std_dev  # type: ignore
                b = (domain_obj.max_value - domain_obj.prior) / std_dev  # type: ignore
                sampled_value = stats.truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=domain_obj.prior,  # type: ignore
                    scale=std_dev,
                )
                if isinstance(domain_obj, Integer):
                    sampled_value = int(round(sampled_value))
                else:
                    sampled_value = float(sampled_value)  # type: ignore
                return cast(T, sampled_value)

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
