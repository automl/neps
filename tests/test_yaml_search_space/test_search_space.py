import pytest

from neps import CategoricalParameter, ConstantParameter, FloatParameter, IntegerParameter
from neps.search_spaces.search_space import (
    SearchSpaceFromYamlFileError,
    pipeline_space_from_yaml,
)


@pytest.mark.yaml_api
def test_correct_yaml_files():
    def test_correct_yaml_file(path):
        """Test the function with a correctly formatted YAML file."""
        pipeline_space = pipeline_space_from_yaml(path)
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
        assert isinstance(pipeline_space["batch_size"], IntegerParameter)
        assert pipeline_space["batch_size"].lower == 100
        assert pipeline_space["batch_size"].upper == 30000
        assert pipeline_space["batch_size"].log is True
        assert pipeline_space["batch_size"].is_fidelity is False
        assert pipeline_space["batch_size"].default is None
        assert pipeline_space["batch_size"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["sec_learning_rate"], FloatParameter)
        assert pipeline_space["sec_learning_rate"].lower == 3.3e-5
        assert pipeline_space["sec_learning_rate"].upper == 0.1
        assert pipeline_space["sec_learning_rate"].log is False
        assert pipeline_space["sec_learning_rate"].is_fidelity is False
        assert pipeline_space["sec_learning_rate"].default is None
        assert pipeline_space["sec_learning_rate"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["parameter_ex"], FloatParameter)
        assert pipeline_space["parameter_ex"].lower == 3.3e-5
        assert pipeline_space["parameter_ex"].upper == 32.0
        assert pipeline_space["parameter_ex"].log is False
        assert pipeline_space["parameter_ex"].is_fidelity is False
        assert pipeline_space["parameter_ex"].default is None
        assert pipeline_space["parameter_ex"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["optimizer"], CategoricalParameter)
        assert pipeline_space["optimizer"].choices == ["adam", "sgd", "rmsprop"]
        assert pipeline_space["optimizer"].is_fidelity is False
        assert pipeline_space["optimizer"].default is None
        assert pipeline_space["optimizer"].default_confidence_score == 2
        assert isinstance(pipeline_space["dropout_rate"], ConstantParameter)
        assert pipeline_space["dropout_rate"].value == 0.5
        assert pipeline_space["dropout_rate"].is_fidelity is False

    test_correct_yaml_file("tests/test_yaml_search_space/correct_config.yaml")
    test_correct_yaml_file(
        "tests/test_yaml_search_space/correct_config_including_types" ".yaml"
    )


@pytest.mark.yaml_api
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


@pytest.mark.yaml_api
def test_incorrect_yaml_file():
    """Test the function with an incorrectly formatted YAML file."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml("tests/test_yaml_search_space/incorrect_config.txt")
    assert str(excinfo.value.exception_type == "ValueError")


@pytest.mark.yaml_api
def test_yaml_file_with_missing_key():
    """Test the function with a YAML file missing a required key."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml("tests/test_yaml_search_space/missing_key_config.yml")
    assert str(excinfo.value.exception_type == "KeyError")


@pytest.mark.yaml_api
def test_yaml_file_with_inconsistent_types():
    """Test the function with a YAML file having inconsistent types for
    'lower' and 'upper'."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/inconsistent_types_config.yml"
        )
    assert str(excinfo.value.exception_type == "TypeError")


@pytest.mark.yaml_api
def test_yaml_file_including_wrong_types():
    """Test the function with a YAML file that defines the wrong but existing type
    int to float as an optional argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/config_including_wrong_types.yaml"
        )
    assert str(excinfo.value.exception_type == "TypeError")


@pytest.mark.yaml_api
def test_yaml_file_including_unkown_types():
    """Test the function with a YAML file that defines an unknown type as an optional
    argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/config_including_unknown_types.yaml"
        )
    assert str(excinfo.value.exception_type == "TypeError")
