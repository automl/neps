from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

import neps
import neps.space.neps_spaces.optimizers.algorithms
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    Operation,
    Pipeline,
    Resampled,
)


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


class DemoHyperparameterSpace(Pipeline):
    float1 = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )


class DemoHyperparameterWithFidelitySpace(Pipeline):
    float1 = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        min_value=-10,
        max_value=10,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Fidelity(
        Integer(
            min_value=1,
            max_value=1000,
        ),
    )


class DemoHyperparameterComplexSpace(Pipeline):
    _small_float = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    _big_float = Float(
        min_value=10,
        max_value=100,
        prior=20,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )

    float1 = Categorical(
        choices=(
            Resampled(_small_float),
            Resampled(_big_float),
        ),
        prior_index=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Categorical(
        choices=(
            Resampled(_small_float),
            Resampled(_big_float),
            float1,
        ),
        prior_index=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    categorical = Categorical(
        choices=(0, 1),
        prior_index=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=1,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer2 = Integer(
        min_value=1,
        max_value=1000,
        prior=10,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        neps.space.neps_spaces.optimizers.algorithms.RandomSearch,
        neps.space.neps_spaces.optimizers.algorithms.ComplexRandomSearch,
    ],
)
def test_hyperparameter_demo(optimizer):
    pipeline_space = DemoHyperparameterSpace()
    root_directory = f"results/hyperparameter_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=neps_space.adjust_evaluation_pipeline_for_neps_space(
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
    [
        neps.space.neps_spaces.optimizers.algorithms.RandomSearch,
        neps.space.neps_spaces.optimizers.algorithms.ComplexRandomSearch,
    ],
)
def test_hyperparameter_with_fidelity_demo(optimizer):
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"results/hyperparameter_with_fidelity_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=neps_space.adjust_evaluation_pipeline_for_neps_space(
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
    [
        neps.space.neps_spaces.optimizers.algorithms.RandomSearch,
        neps.space.neps_spaces.optimizers.algorithms.ComplexRandomSearch,
    ],
)
def test_hyperparameter_complex_demo(optimizer):
    pipeline_space = DemoHyperparameterComplexSpace()
    root_directory = f"results/hyperparameter_complex_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=neps_space.adjust_evaluation_pipeline_for_neps_space(
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


class DemoOperationSpace(Pipeline):
    """A demonstration of how to use operations in a search space.
    This space defines a model that can be optimized using different inner functions
    and a factor. The model can be used to evaluate a set of values and return an objective to minimize.
    """

    # The way to sample `factor` values
    _factor = Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )

    # Sum
    # Will be equivalent to something like
    #   `Sum()`
    # Could have also been defined using the python `sum` function as
    #   `_sum = space.Operation(operator=lambda: sum)`
    _sum = Operation(operator=Sum)

    # MultipliedSum
    # Will be equivalent to something like
    #   `MultipliedSum(factor=0.2)`
    _multiplied_sum = Operation(
        operator=MultipliedSum,
        kwargs={"factor": Resampled(_factor)},
    )

    # Model
    # Will be equivalent to something like one of
    #   `Model(Sum(), factor=0.1)`
    #   `Model(MultipliedSum(factor=0.2), factor=0.1)`
    _inner_function = Categorical(
        choices=(_sum, _multiplied_sum),
    )
    model = Operation(
        operator=Model,
        args=(_inner_function,),
        kwargs={"factor": Resampled(_factor)},
    )

    # An additional hyperparameter
    some_hp = Categorical(
        choices=("hp1", "hp2"),
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        neps.space.neps_spaces.optimizers.algorithms.RandomSearch,
        neps.space.neps_spaces.optimizers.algorithms.ComplexRandomSearch,
    ],
)
def test_operation_demo(optimizer):
    pipeline_space = DemoOperationSpace()
    root_directory = f"results/operation_demo__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=neps_space.adjust_evaluation_pipeline_for_neps_space(
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
