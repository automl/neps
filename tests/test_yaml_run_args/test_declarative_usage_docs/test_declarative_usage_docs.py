import pytest
import os
import neps
from run_pipeline import run_pipeline_constant
BASE_PATH = "tests/test_yaml_run_args/test_declarative_usage_docs/"


@pytest.mark.neps_api
@pytest.mark.parametrize("yaml_file", [
    "simple_example_including_run_pipeline.yaml",
    "full_configuration_template.yaml"
])
def test_run_with_yaml(yaml_file: str) -> None:
    """Test "neps.run" with various run_args.yaml settings to simulate loading options
    for variables."""
    yaml_path = os.path.join(BASE_PATH, yaml_file)
    assert os.path.exists(yaml_path), f"{yaml_file} does not exist."

    try:
        neps.run(run_args=yaml_path)
    except Exception as e:
        pytest.fail(f"NePS run failed for configuration: {yaml_file} with error: {str(e)}"
                    )


@pytest.mark.neps_api
def test_run_with_yaml_and_run_pipeline() -> None:
    """Test "neps.run" with simple_example.yaml as run_args + a run_pipeline that is
    provided separately"""
    yaml_path = os.path.join(BASE_PATH, "simple_example.yaml")
    assert os.path.exists(yaml_path), f"{yaml_path} does not exist."

    try:
        neps.run(run_args=yaml_path, run_pipeline=run_pipeline_constant)
    except Exception as e:
        pytest.fail(f"NePS run failed for configuration: {yaml_path} with error: {str(e)}"
                    )



