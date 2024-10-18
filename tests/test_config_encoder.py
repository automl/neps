import torch
import pytest
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import (
    CategoricalToIntegerTransformer,
    ConfigEncoder,
    MinMaxNormalizer,
)
from neps.search_spaces.hyperparameters import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
)


def test_config_encoder_default() -> None:
    parameters = {
        "a": CategoricalParameter(["cat", "mouse", "dog"]),
        "b": IntegerParameter(5, 6),
        "c": FloatParameter(5, 6),
    }

    encoder = ConfigEncoder.from_parameters(parameters)

    # Numericals first, alphabetic
    # Categoricals last, alphabetic
    assert encoder.transformers == {
        "b": MinMaxNormalizer(parameters["b"].domain),
        "c": MinMaxNormalizer(parameters["c"].domain),
        "a": CategoricalToIntegerTransformer(parameters["a"].choices),
    }

    # Domains, (of each column) match those of the transformers
    assert encoder.domains == [
        Domain.unit_float(),
        Domain.unit_float(),
        Domain.indices(n=len(parameters["a"].choices), is_categorical=True),
    ]

    assert encoder.ncols == len(parameters)
    assert encoder.n_numerical == 2
    assert encoder.n_categorical == 1
    assert encoder.numerical_slice == slice(0, 2)
    assert encoder.categorical_slice == slice(2, 3)
    assert encoder.index_of == {"a": 2, "b": 0, "c": 1}
    assert encoder.domain_of == {
        "b": Domain.unit_float(),
        "c": Domain.unit_float(),
        "a": Domain.indices(n=len(parameters["a"].choices), is_categorical=True),
    }
    assert encoder.constants == {}

    configs = [
        {"c": 5.5, "b": 5, "a": "cat"},
        {"c": 5.5, "b": 5, "a": "dog"},
        {"c": 6, "b": 6, "a": "mouse"},
    ]
    encoded = encoder.encode(configs)
    expcected_encoding = torch.tensor(
        [
            # b,   c,   a
            [0.0, 0.5, 0.0],  # config 1
            [0.0, 0.5, 2.0],  # config 2
            [1.0, 1.0, 1.0],  # config 3
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(encoded, expcected_encoding, check_dtype=True)

    decoded = encoder.decode(encoded)
    assert decoded == configs


def test_config_encoder_accepts_custom_transformers() -> None:
    parameters = {
        "b": IntegerParameter(5, 6),
        "a": FloatParameter(5, 6),
        "c": CategoricalParameter(["cat", "mouse", "dog"]),
    }
    encoder = ConfigEncoder.from_parameters(
        parameters,
        custom_transformers={
            "c": CategoricalToIntegerTransformer(parameters["c"].choices)
        },
    )
    assert encoder.transformers["c"] == CategoricalToIntegerTransformer(
        parameters["c"].choices
    )


def test_config_encoder_removes_constants_in_encoding_and_includes_in_decoding() -> None:
    parameters = {
        "b": IntegerParameter(5, 6),
        "a": FloatParameter(5, 6),
        "c": CategoricalParameter(["cat", "mouse", "dog"]),
    }

    x = "raspberry"

    encoder = ConfigEncoder.from_parameters(parameters, constants={"x": x})
    assert encoder.constants == {"x": x}

    enc_x = encoder.encode([{"a": 5.5, "b": 5, "c": "cat", "x": x}])

    assert enc_x.shape == (1, 3)  # No x, just a, b, c

    dec_x = encoder.decode(enc_x)
    assert dec_x == [{"a": 5.5, "b": 5, "c": "cat", "x": x}]

    # This doesn't have to hold true, but it's our current behaviour, we could make
    # weaker gaurantees but then we'd have to clone the constants, even if it's very large
    assert dec_x[0]["x"] is x


def test_config_encoder_complains_if_missing_entry_in_config() -> None:
    parameters = {
        "b": IntegerParameter(5, 6),
        "a": FloatParameter(5, 6),
        "c": CategoricalParameter(["cat", "mouse", "dog"]),
    }

    encoder = ConfigEncoder.from_parameters(parameters)

    with pytest.raises(KeyError):
        encoder.encode([{"a": 5.5, "b": 5}])


def test_config_encoder_sorts_parameters_by_name_for_consistent_ordering() -> None:
    parameters = {
        "a": CategoricalParameter([0, 1]),
        "b": IntegerParameter(0, 1),
        "c": FloatParameter(0, 1),
    }
    p1 = dict(sorted(parameters.items()))
    p2 = dict(sorted(parameters.items(), reverse=True))

    encoder_1 = ConfigEncoder.from_parameters(p1)
    encoder_2 = ConfigEncoder.from_parameters(p2)

    assert encoder_1.index_of["a"] == 2
    assert encoder_1.index_of["b"] == 0
    assert encoder_1.index_of["c"] == 1

    assert encoder_2.index_of["a"] == 2
    assert encoder_2.index_of["b"] == 0
    assert encoder_2.index_of["c"] == 1
