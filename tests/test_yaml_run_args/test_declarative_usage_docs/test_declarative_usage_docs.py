import pytest
import subprocess
import os
import sys
BASE_PATH = "tests/test_yaml_run_args/test_declarative_usage_docs/"


@pytest.mark.neps_api
@pytest.mark.parametrize("yaml_file", [
    "simple_example_including_run_pipeline.yaml",
    "full_configuration_template.yaml"
])
def test_run_with_yaml(yaml_file: str) -> None:
    """Test "neps.run" with various run_args.yaml settings to simulate loading options
    for variables."""
    assert os.path.exists(BASE_PATH + yaml_file), f"{yaml_file} does not exist."

    try:
        subprocess.check_call([sys.executable, BASE_PATH + 'neps_run.py', BASE_PATH +
                               yaml_file])
    except subprocess.CalledProcessError:
        pytest.fail(f"NePS run failed for configuration: {yaml_file}")
