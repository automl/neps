"""This module defines various samplers for NEPS spaces, allowing for different sampling
strategies such as predefined values, random sampling, and mutation-based sampling.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from neps.space.neps_spaces.neps_space import (
    ConfidenceLevel,
    Domain,
    DomainSampler,
    Pipeline,
)

T = TypeVar("T")
P = TypeVar("P", bound="Pipeline")


class OnlyPredefinedValuesSampler(DomainSampler):
    """A sampler that only returns predefined values for a given path.
    If the path is not found in the predefined values, it raises a ValueError.
    :param predefined_samplings: A mapping of paths to predefined values.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        """Initialize the sampler with predefined samplings.
        :param predefined_samplings: A mapping of paths to predefined values.
        :raises ValueError: If predefined_samplings is empty.
        """
        self._predefined_samplings = predefined_samplings

    def __call__(
        self,
        *,
        domain_obj: Domain[T],  # noqa: ARG002
        current_path: str,
    ) -> T:
        """Sample a value from the predefined samplings for the given path.
        :param domain_obj: The domain object, not used in this sampler.
        :param current_path: The path for which to sample a value.
        :return: The predefined value for the given path.
        :raises ValueError: If the current path is not in the predefined samplings.
        """
        if current_path not in self._predefined_samplings:
            raise ValueError(f"No predefined value for path: {current_path!r}.")
        return cast(T, self._predefined_samplings[current_path])


class RandomSampler(DomainSampler):
    """A sampler that randomly samples from a predefined set of values.
    If the current path is not in the predefined values, it samples from the domain.
    :param predefined_samplings: A mapping of paths to predefined values.
    This sampler will use these values if available, otherwise it will sample from the
    domain.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
    ):
        """Initialize the sampler with predefined samplings.
        :param predefined_samplings: A mapping of paths to predefined values.
        :raises
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
        :param domain_obj: The domain object from which to sample.
        :param current_path: The path for which to sample a value.
        :return: A sampled value, either from the predefined samplings or from the
        domain.
        :raises ValueError: If the current path is not in the predefined samplings and
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
    :param fallback_sampler: A DomainSampler to use if the prior is not available.
    :param prior_use_probability: The probability of using the prior value when
    available.
    This should be a float between 0 and 1, where 0 means never use the prior and 1 means
    always use it.
    :raises ValueError: If the prior_use_probability is not between 0 and 1.
    """

    def __init__(
        self,
        fallback_sampler: DomainSampler,
        prior_use_probability: float,
    ):
        """Initialize the sampler with a fallback sampler and a prior use probability.
        :param fallback_sampler: A DomainSampler to use if the prior is not available.
        :param prior_use_probability: The probability of using the prior value when
        available.
        This should be a float between 0 and 1, where 0 means never use the prior and 1
        means always use it.
        :raises ValueError: If the prior_use_probability is not between 0 and 1.
        """
        if not 0 <= prior_use_probability <= 1:
            raise ValueError(
                "The given `prior_use_probability` value is out of range:"
                f" {prior_use_probability!r}."
            )

        self._fallback_sampler = fallback_sampler
        self._prior_use_probability = prior_use_probability

    def __call__(
        self,
        *,
        domain_obj: Domain[T],
        current_path: str,
    ) -> T:
        """Sample a value from the domain, using the prior if available and according to
        the prior use probability.
        :param domain_obj: The domain object from which to sample.
        :param current_path: The path for which to sample a value.
        :return: A sampled value, either from the prior or from the fallback sampler.
        :raises ValueError: If the domain does not have a prior defined and the fallback
        sampler is not provided.
        """
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
    :param predefined_samplings: A mapping of paths to predefined values.
    :param n_forgets: The number of predefined samplings to forget.
    This should be an integer greater than 0 and less than or equal to the number of
    predefined samplings.
    :raises ValueError: If n_forgets is not a valid integer or if it exceeds the number
    of predefined samplings.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_forgets: int,
    ):
        """Initialize the sampler with predefined samplings and a number of forgets.
        :param predefined_samplings: A mapping of paths to predefined values.
        :param n_forgets: The number of predefined samplings to forget.
        This should be an integer greater than 0 and less than or equal to the number of
        predefined samplings.
        :raises ValueError: If n_forgets is not a valid integer or if it exceeds the
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
        :param domain_obj: The domain object from which to sample.
        :param current_path: The path for which to sample a value.
        :return: A sampled value, either from the mutated predefined samplings or from
        the domain.
        :raises ValueError: If the current path is not in the mutated predefined
        samplings and the domain does not have a prior defined.
        """
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)


class MutatateUsingCentersSampler(DomainSampler):
    """A sampler that mutates predefined samplings by forgetting a certain number of them,
    but still uses the original values as centers for sampling.
    :param predefined_samplings: A mapping of paths to predefined values.
    :param n_mutations: The number of predefined samplings to mutate.
    This should be an integer greater than 0 and less than or equal to the number of
    predefined samplings.
    :raises ValueError: If n_mutations is not a valid integer or if it exceeds the number
    of predefined samplings.
    """

    def __init__(
        self,
        predefined_samplings: Mapping[str, Any],
        n_mutations: int,
    ):
        """Initialize the sampler with predefined samplings and a number of mutations.
        :param predefined_samplings: A mapping of paths to predefined values.
        :param n_mutations: The number of predefined samplings to mutate.
        This should be an integer greater than 0 and less than or equal to the number of
        predefined samplings.
        :raises ValueError: If n_mutations is not a valid integer or if it exceeds
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
        :param domain_obj: The domain object from which to sample.
        :param current_path: The path for which to sample a value.
        :return: A sampled value, either from the kept samplings or from the domain,
        using the original values as centers if necessary.
        :raises ValueError: If the current path is not in the kept samplings and the
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
    probability-based
    selection of values from either source.
    :param predefined_samplings_1: The first set of predefined samplings.
    :param predefined_samplings_2: The second set of predefined samplings.
    :param prefer_first_probability: The probability of preferring values from the first
    set over the second set when both have values for the same path.
    This should be a float between 0 and 1, where 0 means always prefer the second set
    and 1 means always prefer the first set.
    :raises ValueError: If prefer_first_probability is not between 0 and 1.
    :raises CrossoverNotPossibleError: If no crossovers were made between the two sets
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
        :param predefined_samplings_1: The first set of predefined samplings.
        :param predefined_samplings_2: The second set of predefined samplings.
        :param prefer_first_probability: The probability of preferring values from the
        first set over the second set when both have values for the same path.
        This should be a float between 0 and 1, where 0 means always prefer the second
        set and 1 means always prefer the first set.
        :raises ValueError: If prefer_first_probability is not between 0 and 1.
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
        :param domain_obj: The domain object from which to sample.
        :param current_path: The path for which to sample a value.
        :return: A sampled value, either from the crossed-over predefined samplings or
        from the domain.
        :raises ValueError: If the current path is not in the crossed-over predefined
        samplings and the domain does not have a prior defined.
        """
        return self._random_sampler(domain_obj=domain_obj, current_path=current_path)
