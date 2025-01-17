from __future__ import annotations

import torch

from neps.space import Categorical, ConfigEncoder, Float, Integer, SearchSpace
from neps.space.functions import pairwise_dist


def test_config_encoder_pdist_calculation() -> None:
    parameters = SearchSpace(
        {
            "a": Categorical(["cat", "mouse", "dog"]),
            "b": Integer(1, 10),
            "c": Float(1, 10),
        }
    )
    encoder = ConfigEncoder.from_space(parameters)
    config1 = {"a": "cat", "b": 1, "c": 1.0}
    config2 = {"a": "mouse", "b": 10, "c": 10.0}

    # Same config, no distance
    x = encoder.encode([config1, config1])
    dist = pairwise_dist(x, encoder=encoder, square_form=False)
    assert dist.item() == 0.0

    # Opposite configs, max distance
    x = encoder.encode([config1, config2])
    dist = pairwise_dist(x, encoder=encoder, square_form=False)

    # The first config should have it's p2 euclidean distance as the norm
    # of the distances between these two configs, i.e. the distance along the
    # diagonal of a unit-square they belong to
    _first_config_numerical_encoding = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    _second_config_numerical_encoding = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    _expected_numerical_dist = torch.linalg.norm(
        _first_config_numerical_encoding - _second_config_numerical_encoding,
        ord=2,
    )

    # The categorical distance should just be one, as they are different
    _expected_categorical_dist = 1.0

    _expected_dist = _expected_numerical_dist + _expected_categorical_dist
    assert torch.isclose(dist, _expected_dist)


def test_config_encoder_pdist_squareform() -> None:
    parameters = SearchSpace(
        {
            "a": Categorical(["cat", "mouse", "dog"]),
            "b": Integer(1, 10),
            "c": Float(1, 10),
        }
    )
    encoder = ConfigEncoder.from_space(parameters)
    config1 = {"a": "cat", "b": 1, "c": 1.0}
    config2 = {"a": "dog", "b": 5, "c": 5}
    config3 = {"a": "mouse", "b": 10, "c": 10.0}

    # Same config, no distance
    x = encoder.encode([config1, config2, config3])
    dist = pairwise_dist(x, encoder=encoder, square_form=False)

    # 3 possible distances
    assert dist.shape == (3,)
    torch.testing.assert_close(
        dist,
        torch.tensor([1.6285, 2.4142, 1.7857], dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    dist_sq = pairwise_dist(x, encoder=encoder, square_form=True)
    assert dist_sq.shape == (3, 3)

    # Distance to self along diagonal should be 0
    torch.testing.assert_close(dist_sq.diagonal(), torch.zeros(3, dtype=torch.float64))

    # Should be symmetric
    torch.testing.assert_close(dist_sq, dist_sq.T)
