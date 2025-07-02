from __future__ import annotations

import re

import pytest

import neps.space.neps_spaces.sampling
from neps.space.neps_spaces import neps_space


class DemoHyperparametersWithFidelitySpace(neps_space.Pipeline):
    constant1: int = 42
    float1: float = neps_space.Float(
        min_value=0,
        max_value=1,
        prior=0.1,
        prior_confidence=neps_space.ConfidenceLevel.MEDIUM,
    )
    fidelity_integer1: int = neps_space.Fidelity(
        domain=neps_space.Integer(
            min_value=1,
            max_value=1000,
        ),
    )


def test_fidelity_creation_raises_when_domain_has_prior():
    # Creating a fidelity object with a domain that has a prior should not be possible.
    with pytest.raises(
        ValueError,
        match=re.escape("The domain of a Fidelity can not have priors: "),
    ):
        neps_space.Fidelity(
            domain=neps_space.Integer(
                min_value=1,
                max_value=1000,
                prior=10,
                prior_confidence=neps_space.ConfidenceLevel.MEDIUM,
            ),
        )


def test_fidelity_resolution_raises_when_resolved_with_no_environment_value():
    pipeline = DemoHyperparametersWithFidelitySpace()

    # Resolve a pipeline which contains a fidelity with an empty environment.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No value is available in the environment for fidelity 'fidelity_integer1'.",
        ),
    ):
        neps_space.resolve(pipeline=pipeline)


def test_fidelity_resolution_raises_when_resolved_with_invalid_value():
    pipeline = DemoHyperparametersWithFidelitySpace()

    # Resolve a pipeline which contains a fidelity,
    # with an environment value for it, that is out of the allowed range.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value for fidelity with name 'fidelity_integer1' is outside its allowed"
            " range [1, 1000]. Received: -10."
        ),
    ):
        neps_space.resolve(
            pipeline=pipeline,
            environment_values={"fidelity_integer1": -10},
        )


def test_fidelity_resolution_works():
    pipeline = DemoHyperparametersWithFidelitySpace()

    # Resolve a pipeline which contains a fidelity,
    # with a valid value for it in the environment.
    resolved_pipeline, resolution_context = neps_space.resolve(
        pipeline=pipeline,
        environment_values={"fidelity_integer1": 10},
    )

    assert resolved_pipeline.constant1 == 42
    assert 0.0 <= resolved_pipeline.float1 <= 1.0
    assert resolved_pipeline.fidelity_integer1 == 10


def test_fidelity_resolution_with_context_works():
    pipeline = DemoHyperparametersWithFidelitySpace()

    samplings_to_make = {
        "Resolvable.float1::float__0_1_False": 0.5,
    }
    environment_values = {
        "fidelity_integer1": 10,
    }

    # Resolve a pipeline which contains a fidelity,
    # with a valid value for it in the environment.
    resolved_pipeline, resolution_context = neps_space.resolve(
        pipeline=pipeline,
        domain_sampler=neps.space.neps_spaces.sampling.OnlyPredefinedValuesSampler(
            predefined_samplings=samplings_to_make,
        ),
        environment_values=environment_values,
    )

    assert resolved_pipeline.constant1 == 42
    assert resolved_pipeline.float1 == 0.5
    assert resolved_pipeline.fidelity_integer1 == 10

    assert resolution_context.samplings_made == samplings_to_make
    assert resolution_context.environment_values == environment_values
