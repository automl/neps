from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

import neps
from neps.space.new_space import space


def hyperparameter_pipeline_to_optimize(
    float1: float,
    float2: float,
    categorical: int,
    integer1: int,
    integer2: int,
):
    assert isinstance(float1, float)
    assert isinstance(float2, float)
    assert isinstance(categorical, int)
    assert isinstance(integer1, int)
    assert isinstance(integer2, int)

    objective_to_minimize = -float(float1 + float2 + categorical + integer1 + integer2)
    assert isinstance(objective_to_minimize, float)

    return objective_to_minimize


class DemoHyperparameterSpace(space.Pipeline):
    float1 = space.Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    float2 = space.Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    categorical = space.Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer1 = space.Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer2 = space.Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )


class DemoHyperparameterWithFidelitySpace(space.Pipeline):
    float1 = space.Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    float2 = space.Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    categorical = space.Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer1 = space.Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer2 = space.Fidelity(
        space.Integer(
            min_value=1,
            max_value=1000,
        ),
    )


class DemoHyperparameterComplexSpace(space.Pipeline):
    _small_float = space.Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    _big_float = space.Float(
        min_value=10,
        max_value=100,
        prior=20,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    float1 = space.Categorical(
        choices=(
            space.Resampled(_small_float),
            space.Resampled(_big_float),
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    float2 = space.Categorical(
        choices=(
            space.Resampled(_small_float),
            space.Resampled(_big_float),
            float1,
        ),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    categorical = space.Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer1 = space.Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )
    integer2 = space.Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )


@pytest.mark.parametrize(
    "optimizer",
    [space.RandomSearch, space.ComplexRandomSearch],
)
def test_hyperparameter_demo(optimizer):
    pipeline_space = DemoHyperparameterSpace()
    root_directory = f"results/hyperparameter_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            hyperparameter_pipeline_to_optimize,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=10,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    "optimizer",
    [space.RandomSearch, space.ComplexRandomSearch],
)
def test_hyperparameter_with_fidelity_demo(optimizer):
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"results/hyperparameter_with_fidelity_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            hyperparameter_pipeline_to_optimize,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=10,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    "optimizer",
    [space.RandomSearch, space.ComplexRandomSearch],
)
def test_hyperparameter_complex_demo(optimizer):
    pipeline_space = DemoHyperparameterComplexSpace()
    root_directory = f"results/hyperparameter_complex_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            hyperparameter_pipeline_to_optimize,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=10,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)


# -----------------------------------------


class Model:
    """A simple model that takes an inner function and a factor,
    multiplies the result of the inner function by the factor.
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


class MultipliedSum:
    """An inner function that sums the values and multiplies the result by a factor."""

    def __init__(self, factor: float):
        """Initialize the multiplied sum with a factor."""
        self.factor = factor

    def __call__(self, values: Sequence[float]) -> float:
        return self.factor * sum(values)


def operation_pipeline_to_optimize(model: Model, some_hp: str):
    assert isinstance(model, Model)
    assert isinstance(model.factor, float)
    assert isinstance(model.inner_function, Sum | MultipliedSum)
    if isinstance(model.inner_function, MultipliedSum):
        assert isinstance(model.inner_function.factor, float)
    assert some_hp in {"hp1", "hp2"}

    values = list(range(1, 21))
    objective_to_minimize = model(values)
    assert isinstance(objective_to_minimize, float)

    return objective_to_minimize


class DemoOperationSpace(space.Pipeline):
    """A demonstration of how to use operations in a search space.
    This space defines a model that can be optimized using different inner functions
    and a factor. The model can be used to evaluate a set of values and return an objective to minimize.
    """

    # The way to sample `factor` values
    _factor = space.Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=space.ConfidenceLevel.MEDIUM,
    )

    # Sum
    # Will be equivalent to something like
    #   `Sum()`
    # Could have also been defined using the python `sum` function as
    #   `_sum = space.Operation(operator=lambda: sum)`
    _sum = space.Operation(operator=Sum)

    # MultipliedSum
    # Will be equivalent to something like
    #   `MultipliedSum(factor=0.2)`
    _multiplied_sum = space.Operation(
        operator=MultipliedSum,
        kwargs={"factor": space.Resampled(_factor)},
    )

    # Model
    # Will be equivalent to something like one of
    #   `Model(Sum(), factor=0.1)`
    #   `Model(MultipliedSum(factor=0.2), factor=0.1)`
    _inner_function = space.Categorical(
        choices=(_sum, _multiplied_sum),
    )
    model = space.Operation(
        operator=Model,
        args=(_inner_function,),
        kwargs={"factor": space.Resampled(_factor)},
    )

    # An additional hyperparameter
    some_hp = space.Categorical(
        choices=("hp1", "hp2"),
    )


@pytest.mark.parametrize(
    "optimizer",
    [space.RandomSearch, space.ComplexRandomSearch],
)
def test_operation_demo(optimizer):
    pipeline_space = DemoOperationSpace()
    root_directory = f"results/operation_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            operation_pipeline_to_optimize,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=10,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)
