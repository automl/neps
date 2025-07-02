from __future__ import annotations

import pytest

import neps.space.neps_spaces.parameters


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max"),
    [
        (neps.space.neps_spaces.parameters.ConfidenceLevel.LOW, (50, 10, 90)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (50, 25, 75)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (50, 40, 60)),
    ],
)
def test_centering_integer(
    confidence_level,
    expected_prior_min_max,
):
    # Construct domains manually and then with priors.
    # They are constructed in a way that after centering they both
    # refer to identical domain ranges.

    int_prior = 50

    int1 = neps.space.neps_spaces.parameters.Integer(
        min_value=1,
        max_value=100,
    )
    int2 = neps.space.neps_spaces.parameters.Integer(
        min_value=1,
        max_value=100,
        prior=int_prior,
        prior_confidence=confidence_level,
    )

    int1_centered = int1.centered_around(int_prior, confidence_level)
    int2_centered = int2.centered_around(int2.prior, int2.prior_confidence)

    assert int_prior == expected_prior_min_max[0]
    assert (
        (
            int1_centered.prior,
            int1_centered.min_value,
            int1_centered.max_value,
        )
        == (
            int2_centered.prior,
            int2_centered.min_value,
            int2_centered.max_value,
        )
        == expected_prior_min_max
    )

    int1_centered.sample()
    int2_centered.sample()


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max"),
    [
        (
            neps.space.neps_spaces.parameters.ConfidenceLevel.LOW,
            (50.0, 10.399999999999999, 89.6),
        ),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (50.0, 25.25, 74.75)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (50.0, 40.1, 59.9)),
    ],
)
def test_centering_float(
    confidence_level,
    expected_prior_min_max,
):
    # Construct domains manually and then with priors.
    # They are constructed in a way that after centering they both
    # refer to identical domain ranges.

    float_prior = 50.0

    float1 = neps.space.neps_spaces.parameters.Float(
        min_value=1.0,
        max_value=100.0,
    )
    float2 = neps.space.neps_spaces.parameters.Float(
        min_value=1.0,
        max_value=100.0,
        prior=float_prior,
        prior_confidence=confidence_level,
    )

    float1_centered = float1.centered_around(float_prior, confidence_level)
    float2_centered = float2.centered_around(float2.prior, float2.prior_confidence)

    assert float_prior == expected_prior_min_max[0]
    assert (
        (
            float1_centered.prior,
            float1_centered.min_value,
            float1_centered.max_value,
        )
        == (
            float2_centered.prior,
            float2_centered.min_value,
            float2_centered.max_value,
        )
        == expected_prior_min_max
    )

    float1_centered.sample()
    float2_centered.sample()


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max_value"),
    [
        (neps.space.neps_spaces.parameters.ConfidenceLevel.LOW, (40, 0, 80, 50)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (25, 0, 50, 50)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (10, 0, 20, 50)),
    ],
)
def test_centering_categorical(
    confidence_level,
    expected_prior_min_max_value,
):
    # Construct domains manually and then with priors.
    # They are constructed in a way that after centering they both
    # refer to identical domain ranges.

    categorical_prior_index_original = 49

    categorical1 = neps.space.neps_spaces.parameters.Categorical(
        choices=tuple(range(1, 101)),
    )
    categorical2 = neps.space.neps_spaces.parameters.Categorical(
        choices=tuple(range(1, 101)),
        prior_index=categorical_prior_index_original,
        prior_confidence=confidence_level,
    )

    categorical1_centered = categorical1.centered_around(
        categorical_prior_index_original, confidence_level
    )
    categorical2_centered = categorical2.centered_around(
        categorical2.prior, categorical2.prior_confidence
    )

    # During the centering of categorical objects, the prior index will change.
    assert categorical_prior_index_original != expected_prior_min_max_value[0]

    assert (
        (
            categorical1_centered.prior,
            categorical1_centered.min_value,
            categorical1_centered.max_value,
            categorical1_centered.choices[categorical1_centered.prior],
        )
        == (
            categorical2_centered.prior,
            categorical2_centered.min_value,
            categorical2_centered.max_value,
            categorical2_centered.choices[categorical2_centered.prior],
        )
        == expected_prior_min_max_value
    )

    categorical1_centered.sample()
    categorical2_centered.sample()


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max"),
    [
        (neps.space.neps_spaces.parameters.ConfidenceLevel.LOW, (10, 5, 13)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (10, 7, 13)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (10, 8, 12)),
    ],
)
def test_centering_stranger_ranges_integer(
    confidence_level,
    expected_prior_min_max,
):
    int1 = neps.space.neps_spaces.parameters.Integer(
        min_value=1,
        max_value=13,
    )
    int1_centered = int1.centered_around(10, confidence_level)

    int2 = neps.space.neps_spaces.parameters.Integer(
        min_value=1,
        max_value=13,
        prior=10,
        prior_confidence=confidence_level,
    )
    int2_centered = int2.centered_around(int2.prior, int2.prior_confidence)

    assert (
        int1_centered.prior,
        int1_centered.min_value,
        int1_centered.max_value,
    ) == expected_prior_min_max
    assert (
        int2_centered.prior,
        int2_centered.min_value,
        int2_centered.max_value,
    ) == expected_prior_min_max

    int1_centered.sample()
    int2_centered.sample()


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max"),
    [
        (
            neps.space.neps_spaces.parameters.ConfidenceLevel.LOW,
            (0.5, 0.09999999999999998, 0.9),
        ),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (0.5, 0.25, 0.75)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (0.5, 0.4, 0.6)),
    ],
)
def test_centering_stranger_ranges_float(
    confidence_level,
    expected_prior_min_max,
):
    float1 = neps.space.neps_spaces.parameters.Float(
        min_value=0.0,
        max_value=1.0,
    )
    float1_centered = float1.centered_around(0.5, confidence_level)

    float2 = neps.space.neps_spaces.parameters.Float(
        min_value=0.0,
        max_value=1.0,
        prior=0.5,
        prior_confidence=confidence_level,
    )
    float2_centered = float2.centered_around(float2.prior, float2.prior_confidence)

    assert (
        float1_centered.prior,
        float1_centered.min_value,
        float1_centered.max_value,
    ) == expected_prior_min_max
    assert (
        float2_centered.prior,
        float2_centered.min_value,
        float2_centered.max_value,
    ) == expected_prior_min_max

    float1_centered.sample()
    float2_centered.sample()


@pytest.mark.parametrize(
    ("confidence_level", "expected_prior_min_max_value"),
    [
        (neps.space.neps_spaces.parameters.ConfidenceLevel.LOW, (2, 0, 5, 2)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM, (2, 0, 4, 2)),
        (neps.space.neps_spaces.parameters.ConfidenceLevel.HIGH, (1, 0, 2, 2)),
    ],
)
def test_centering_stranger_ranges_categorical(
    confidence_level,
    expected_prior_min_max_value,
):
    categorical1 = neps.space.neps_spaces.parameters.Categorical(
        choices=tuple(range(7)),
    )
    categorical1_centered = categorical1.centered_around(2, confidence_level)

    categorical2 = neps.space.neps_spaces.parameters.Categorical(
        choices=tuple(range(7)),
        prior_index=2,
        prior_confidence=confidence_level,
    )
    categorical2_centered = categorical2.centered_around(
        categorical2.prior, categorical2.prior_confidence
    )

    assert (
        categorical1_centered.prior,
        categorical1_centered.min_value,
        categorical1_centered.max_value,
        categorical1_centered.choices[categorical1_centered.prior],
    ) == expected_prior_min_max_value

    assert (
        categorical2_centered.prior,
        categorical2_centered.min_value,
        categorical2_centered.max_value,
        categorical2_centered.choices[categorical2_centered.prior],
    ) == expected_prior_min_max_value

    categorical1_centered.sample()
    categorical2_centered.sample()
