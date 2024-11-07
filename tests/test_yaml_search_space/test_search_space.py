from pathlib import Path

import pytest
from neps.search_spaces.search_space import (
    SearchSpaceFromYamlFileError,
    pipeline_space_from_yaml,
)

from neps import Categorical, Constant, Float, Integer

BASE_PATH = "tests/test_yaml_search_space/"


@pytest.mark.neps_api
def test_correct_yaml_files():
    def test_correct_yaml_file(path):
        """Test the function with a correctly formatted YAML file."""
        pipeline_space = pipeline_space_from_yaml(path)
        assert isinstance(pipeline_space, dict)
        float1 = Float(0.00001, 0.1, log=True, is_fidelity=False)
        assert float1.__eq__(pipeline_space["param_float1"]) is True
        int1 = Integer(3, 30, log=False, is_fidelity=True)
        assert int1.__eq__(pipeline_space["param_int1"]) is True
        int2 = Integer(100, 30000, log=True, is_fidelity=False)
        assert int2.__eq__(pipeline_space["param_int2"]) is True
        float2 = Float(3.3e-5, 0.15, log=False)
        assert float2.__eq__(pipeline_space["param_float2"]) is True
        cat1 = Categorical([2, "sgd", 10e-3])
        assert cat1.__eq__(pipeline_space["param_cat"]) is True
        const1 = Constant(0.5)
        assert const1.__eq__(pipeline_space["param_const1"]) is True
        const2 = Constant(1e3)
        assert const2.__eq__(pipeline_space["param_const2"]) is True

    test_correct_yaml_file(BASE_PATH + "correct_config.yaml")
    test_correct_yaml_file(BASE_PATH + "correct_config_including_types.yaml")


@pytest.mark.neps_api
def test_correct_including_priors_yaml_file():
    """Test the function with a correctly formatted YAML file."""
    pipeline_space = pipeline_space_from_yaml(
        BASE_PATH + "correct_config_including_priors.yml"
    )
    assert isinstance(pipeline_space, dict)
    float1 = Float(0.00001, 0.1, log=True, is_fidelity=False, default=3.3e-2, default_confidence="high")
    assert float1.__eq__(pipeline_space["learning_rate"]) is True
    int1 = Integer(3, 30, log=False, is_fidelity=True)
    assert int1.__eq__(pipeline_space["num_epochs"]) is True
    cat1 = Categorical(["adam", 90e-3, "rmsprop"], default=90e-3, default_confidence="medium")
    assert cat1.__eq__(pipeline_space["optimizer"]) is True
    const1 = Constant(1e3)
    assert const1.__eq__(pipeline_space["dropout_rate"]) is True


@pytest.mark.neps_api
def test_incorrect_yaml_file():
    """Test the function with an incorrectly formatted YAML file."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(Path(BASE_PATH + "incorrect_config.txt"))
    assert excinfo.value.exception_type == "ValueError"


@pytest.mark.neps_api
def test_yaml_file_with_missing_key():
    """Test the function with a YAML file missing a required key."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "missing_key_config.yml")
    assert excinfo.value.exception_type == "KeyError"


@pytest.mark.neps_api
def test_yaml_file_with_inconsistent_types():
    """Test the function with a YAML file having inconsistent types for
    'lower' and 'upper'."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "inconsistent_types_config.yml")
    assert str(excinfo.value.exception_type == "TypeError")
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(Path(BASE_PATH + "inconsistent_types_config2.yml"))
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_wrong_types():
    """Test the function with a YAML file that defines the wrong but existing type
    int to float as an optional argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(Path(BASE_PATH + "inconsistent_types_config2.yml"))
        assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_unkown_types():
    """Test the function with a YAML file that defines an unknown type as an optional
    argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "config_including_unknown_types.yaml")
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_including_not_allowed_parameter_keys():
    """Test the function with a YAML file that defines an unknown type as an optional
    argument"""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "not_allowed_key_config.yml")
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_yaml_file_default_parameter_not_in_range():
    """Test if the default value outside the specified range is
    correctly identified and handled."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "default_not_in_range_config.yaml")
    assert excinfo.value.exception_type == "ValueError"


@pytest.mark.neps_api
def test_float_log_not_boolean():
    """Test if an exception is raised when the 'log' attribute is not a boolean."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "not_boolean_type_log_config.yaml")
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_float_is_fidelity_not_boolean():
    """Test if an exception is raised when for Float the 'is_fidelity'
    attribute is not a boolean."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(
            BASE_PATH + "not_boolean_type_is_fidelity_float_config.yaml"
        )
    assert excinfo.value.exception_type == "TypeError"


@pytest.mark.neps_api
def test_categorical_default_value_not_in_choices():
    """Test if a ValueError is raised when the default value is not in the choices
    for a Categorical."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "default_value_not_in_choices_config.yaml")
    assert excinfo.value.exception_type == "ValueError"

@pytest.mark.neps_api
def test_incorrect_fidelity_parameter_bounds():
    """Test if a ValueError is raised when the bounds of a fidelity parameter are
    not correctly specified."""
    with pytest.raises(SearchSpaceFromYamlFileError) as excinfo:
        pipeline_space_from_yaml(BASE_PATH + "incorrect_fidelity_bounds_config.yaml")
    assert excinfo.value.exception_type == "ValueError"
