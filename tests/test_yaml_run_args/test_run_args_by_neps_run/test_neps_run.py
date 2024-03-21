import pytest
import subprocess
import os
import sys
BASE_PATH = "tests/test_yaml_run_args/test_run_args_by_neps_run/"


@pytest.mark.neps_api
@pytest.mark.parametrize("yaml_file", [
    "config.yaml",
    "loading_pipeline_space.yaml",
    "loading_optimizer.yaml"
])
def test_run_with_yaml(yaml_file):
    """Test "neps.run" with various run_args.yaml settings to simulate loading options
    for variables."""
    assert os.path.exists(BASE_PATH + yaml_file), f"{yaml_file} does not exist."

    try:
        subprocess.check_call([sys.executable, BASE_PATH + 'neps_run.py', BASE_PATH +
                               yaml_file])
    except subprocess.CalledProcessError:
        pytest.fail(f"NePS run failed for configuration: {yaml_file}")
