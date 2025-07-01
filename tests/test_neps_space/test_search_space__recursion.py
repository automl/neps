from __future__ import annotations

from collections.abc import Callable, Sequence

from neps.space.new_space import space


class Model:
    """An inner function that sums the values and multiplies the result by a factor.
    This class can be recursively used in a search space to create nested models.
    """

    def __init__(
        self,
        inner_function: Callable[[Sequence[float]], float],
        factor: float,
    ):
        """Initialize the model with an inner function and a factor."""
        self.inner_function = inner_function
        self.factor = factor

    def __call__(self, values: Sequence[float]) -> float:
        return self.factor * self.inner_function(values)


class Sum:
    """A simple inner function that sums the values."""

    def __call__(self, values: Sequence[float]) -> float:
        return sum(values)


class DemoRecursiveOperationSpace(space.Pipeline):
    # The way to sample `factor` values
    _factor = space.Float(min_value=0, max_value=1)

    # Sum
    _sum = space.Operation(operator=Sum)

    # Model
    # Can recursively request itself as an arg.
    # Will be equivalent to something like one of
    #   `Model(Sum(), factor=0.1)`
    #   `Model(Model(Sum(), factor=0.1), factor=0.1)`
    #   `Model(Model(Model(Sum(), factor=0.1), factor=0.1), factor=0.1)`
    #   ...
    # If we want the `factor` values to be different,
    # we just request a resample for them
    _inner_function = space.Categorical(
        choices=(_sum, space.Resampled("model")),
    )
    model = space.Operation(
        operator=Model,
        args=(space.Resampled(_inner_function),),
        kwargs={"factor": _factor},
    )


def test_recursion():
    pipeline = DemoRecursiveOperationSpace()

    # Across `n` iterations we collect the number of seen inner `Model` counts.
    # We expect to see at least `k` cases for that number
    expected_minimal_number_of_recursions = 3
    seen_inner_model_counts = []

    for _ in range(200):
        resolved_pipeline, _resolution_context = space.resolve(pipeline)

        model = resolved_pipeline.model
        assert model.operator is Model

        inner_function = model.args[0]
        seen_factors = [model.kwargs["factor"]]
        seen_inner_model_count = 0

        # Loop into the inner operators until we have no more nested `Model` args
        while inner_function.operator is Model:
            seen_factors.append(inner_function.kwargs["factor"])
            seen_inner_model_count += 1
            inner_function = inner_function.args[0]

        # At this point we should have gone deep enough to have the terminal `Sum`
        assert inner_function.operator is Sum

        # We should have seen as many factors as inner models + 1 for the outer one
        assert len(seen_factors) == seen_inner_model_count + 1

        # All the factors should be the same value (shared)
        assert len(set(seen_factors)) == 1
        assert isinstance(seen_factors[0], float)

        # Add the number of seen `Model` operator in the loop
        seen_inner_model_counts.append(seen_inner_model_count)

    assert len(set(seen_inner_model_counts)) >= expected_minimal_number_of_recursions


# TODO: test context with recursion (`samplings_to_make`)
