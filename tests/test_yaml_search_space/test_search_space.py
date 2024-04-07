from pathlib import Path

import pytest
from neps.search_spaces.search_space import (
    SearchSpaceFromYamlFileError,
    pipeline_space_from_yaml,
)

from neps import CategoricalParameter, ConstantParameter, FloatParameter, IntegerParameter


@pytest.mark.neps_api
def test_correct_yaml_files():
    def test_correct_yaml_file(path):
        """Test the function with a correctly formatted YAML file."""
        pipeline_space = pipeline_space_from_yaml(path)
        assert isinstance(pipeline_space, dict)
        assert isinstance(pipeline_space["param_float1"], FloatParameter)
        assert pipeline_space["param_float1"].lower == 0.00001
        assert pipeline_space["param_float1"].upper == 0.1
        assert pipeline_space["param_float1"].log is True
        assert pipeline_space["param_float1"].is_fidelity is False
        assert pipeline_space["param_float1"].default is None
        assert pipeline_space["param_float1"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["param_int1"], IntegerParameter)
        assert pipeline_space["param_int1"].lower == -3
        assert pipeline_space["param_int1"].upper == 30
        assert pipeline_space["param_int1"].log is False
        assert pipeline_space["param_int1"].is_fidelity is True
        assert pipeline_space["param_int1"].default is None
        assert pipeline_space["param_int1"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["param_int2"], IntegerParameter)
        assert pipeline_space["param_int2"].lower == 100
        assert pipeline_space["param_int2"].upper == 30000
        assert pipeline_space["param_int2"].log is True
        assert pipeline_space["param_int2"].is_fidelity is False
        assert pipeline_space["param_int2"].default is None
        assert pipeline_space["param_int2"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["param_float2"], FloatParameter)
        assert pipeline_space["param_float2"].lower == 3.3e-5
        assert pipeline_space["param_float2"].upper == 0.15
        assert pipeline_space["param_float2"].log is False
        assert pipeline_space["param_float2"].is_fidelity is False
        assert pipeline_space["param_float2"].default is None
        assert pipeline_space["param_float2"].default_confidence_score == 0.5
        assert isinstance(pipeline_space["param_cat"], CategoricalParameter)
        assert pipeline_space["param_cat"].choices == [2, "sgd", 10e-3]
        assert pipeline_space["param_cat"].is_fidelity is False
        assert pipeline_space["param_cat"].default is None
        assert pipeline_space["param_cat"].default_confidence_score == 2
        assert isinstance(pipeline_space["param_const1"], ConstantParameter)
        assert pipeline_space["param_const1"].value == 0.5
        assert pipeline_space["param_const1"].is_fidelity is False
        assert isinstance(pipeline_space["param_const2"], ConstantParameter)
        assert pipeline_space["param_const2"].value == 1e3
        assert pipeline_space["param_const2"].is_fidelity is True

    test_correct_yaml_file("tests/test_yaml_search_space/correct_config.yaml")
    test_correct_yaml_file(
        "tests/test_yaml_search_space/correct_config_including_types" ".yaml"
    )


@pytest.mark.neps_api
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
    assert pipeline_space["learning_rate"].default == 3.3e-2
    assert pipeline_space["learning_rate"].default_confidence_score == 0.125
    assert isinstance(pipeline_space["num_epochs"], IntegerParameter)
    assert pipeline_space["num_epochs"].lower == 3
    assert pipeline_space["num_epochs"].upper == 30
    assert pipeline_space["num_epochs"].log is False
    assert pipeline_space["num_epochs"].is_fidelity is True
    assert pipeline_space["num_epochs"].default == 10
    assert pipeline_space["num_epochs"].default_confidence_score == 0.5
    assert isinstance(pipeline_space["optimizer"], CategoricalParameter)
    assert pipeline_space["optimizer"].choices == ["adam", 90e-3, "rmsprop"]
    assert pipeline_space["optimizer"].is_fidelity is False
    assert pipeline_space["optimizer"].default == 90e-3
    assert pipeline_space["optimizer"].default_confidence_score == 4
    assert isinstance(pipeline_space["dropout_rate"], ConstantParameter)
    assert pipeline_space["dropout_rate"].value == 1e3
    assert pipeline_space["dropout_rate"].default == 1e3


@pytest.mark.neps_api
def test_incorrect_yaml_file():
    """Test the function with an incorrectly formatted YAML file."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            Path("tests/test_yaml_search_space/incorrect_config.txt")
        )
    assert excinfo.value.exception_type == "ValueError"


@pytest.mark.neps_api
def test_yaml_file_with_missing_key():
    """Test the function with a YAML file missing a required key."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml("tests/test_yaml_search_space/missing_key_config.yml")
    assert excinfo.value.exception_type == "KeyError"


@pytest.mark.neps_api
def test_yaml_file_with_inconsistent_types():
    """Test the function with a YAML file having inconsistent types for
    'lower' and 'upper'."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/inconsistent_types_config.yml"
        )
    assert str(excinfo.value.exception_type == "TypeError")
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            Path("tests/test_yaml_search_space/inconsistent_types_config2.yml")
        )
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_wrong_types():
    """Test the function with a YAML file that defines the wrong but existing type
    int to float as an optional argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/config_including_wrong_types.yaml"
        )
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_unkown_types():
    """Test the function with a YAML file that defines an unknown type as an optional
    argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/config_including_unknown_types.yaml"
        )
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_not_allowed_parameter_keys():
    """Test the function with a YAML file that defines an unknown type as an optional
    argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            "tests/test_yaml_search_space/not_allowed_key_config.yml"
        )
    assert excinfo.value.exception_type == "KeyError"
