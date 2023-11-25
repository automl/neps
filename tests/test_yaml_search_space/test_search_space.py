import pytest

from neps import CategoricalParameter, ConstantParameter, FloatParameter, IntegerParameter
from neps.search_spaces.search_space import pipeline_space_from_yaml


@pytest.mark.yaml_search_space
def test_correct_yaml_file():
    """Test the function with a correctly formatted YAML file."""
    pipeline_space = pipeline_space_from_yaml(
        "tests/test_yaml_search_space/correct_config.yaml"
    )
    assert isinstance(pipeline_space, dict)
    assert isinstance(pipeline_space["learning_rate"], FloatParameter)
    assert pipeline_space["learning_rate"].lower == 0.00001
    assert pipeline_space["learning_rate"].upper == 0.1
    assert pipeline_space["learning_rate"].log is True
    assert pipeline_space["optimizer"].is_fidelity is False
    assert pipeline_space["learning_rate"].default is None
    assert pipeline_space["learning_rate"].default_confidence_score == 0.5
    assert isinstance(pipeline_space["num_epochs"], IntegerParameter)
    assert pipeline_space["num_epochs"].lower == 3
    assert pipeline_space["num_epochs"].upper == 30
    assert pipeline_space["num_epochs"].log is False
    assert pipeline_space["num_epochs"].is_fidelity is True
    assert pipeline_space["num_epochs"].default is None
    assert pipeline_space["num_epochs"].default_confidence_score == 0.5
    assert isinstance(pipeline_space["optimizer"], CategoricalParameter)
    assert pipeline_space["optimizer"].choices == ["adam", "sgd", "rmsprop"]
    assert pipeline_space["optimizer"].is_fidelity is False
    assert pipeline_space["optimizer"].default is None
    assert pipeline_space["optimizer"].default_confidence_score == 2
    assert isinstance(pipeline_space["dropout_rate"], ConstantParameter)
    assert pipeline_space["dropout_rate"].value == 0.5
    assert pipeline_space["dropout_rate"].is_fidelity is False


@pytest.mark.yaml_search_space
def test_correct_including_priors_yaml_file():
    """Test the function with a correctly formatted YAML file."""
    pipeline_space = pipeline_space_from_yaml(
        "tests/test_yaml_search_space/correct_config_including_priors.yml"
    )
    assert isinstance(pipeline_space, dict)
    assert isinstance(pipeline_space["learning_rate"], FloatParameter)
    assert pipeline_space["learning_rate"].lower == 0.00001
    assert pipeline_space["learning_rate"].upper == 0.1
    assert pipeline_space["learning_rate"].log is True
    assert pipeline_space["learning_rate"].is_fidelity is False
    assert pipeline_space["learning_rate"].default == 0.001
    assert pipeline_space["learning_rate"].default_confidence_score == 0.125
    assert isinstance(pipeline_space["num_epochs"], IntegerParameter)
    assert pipeline_space["num_epochs"].lower == 3
    assert pipeline_space["num_epochs"].upper == 30
    assert pipeline_space["num_epochs"].log is False
    assert pipeline_space["num_epochs"].is_fidelity is True
    assert pipeline_space["num_epochs"].default == 10
    assert pipeline_space["num_epochs"].default_confidence_score == 0.25
    assert isinstance(pipeline_space["optimizer"], CategoricalParameter)
    assert pipeline_space["optimizer"].choices == ["adam", "sgd", "rmsprop"]
    assert pipeline_space["optimizer"].is_fidelity is False
    assert pipeline_space["optimizer"].default == "sgd"
    assert pipeline_space["optimizer"].default_confidence_score == 4
    assert isinstance(pipeline_space["dropout_rate"], ConstantParameter)
    assert pipeline_space["dropout_rate"].value == 0.5
    assert pipeline_space["dropout_rate"].is_fidelity is True


@pytest.mark.yaml_search_space
def test_incorrect_yaml_file():
    """Test the function with an incorrectly formatted YAML file."""
    with pytest.raises(ValueError):
        pipeline_space_from_yaml("tests/test_yaml_search_space/incorrect_config.txt")


@pytest.mark.yaml_search_space
def test_yaml_file_with_missing_key():
    """Test the function with a YAML file missing a required key."""
    with pytest.raises(KeyError):
        pipeline_space_from_yaml("tests/test_yaml_search_space/missing_key_config.yml")


@pytest.mark.yaml_search_space
def test_yaml_file_with_inconsistent_types():
    """Test the function with a YAML file having inconsistent types for
    'lower' and 'upper'."""
    with pytest.raises(TypeError):
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/inconsistent_types_config.yml"
        )
