import pytest
import os
import subprocess
import sys
from pathlib import Path

BASE_PATH = Path("tests") / "test_yaml_run_args" / "test_declarative_usage_docs"


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "yaml_file",
    [
        "simple_example_including_run_pipeline.yaml",
        "full_configuration_template.yaml",
        "defining_hooks.yaml",
        "customizing_neps_optimizer.yaml",
        "loading_own_optimizer.yaml",
        "loading_pipeline_space_dict.yaml",
        "outsourcing_optimizer.yaml",
        "outsourcing_pipeline_space.yaml",
    ],
)
def test_run_with_yaml(yaml_file: str) -> None:
    """
    Test 'neps.run' with various run_args.yaml settings to simulate loading options
    for variables.
    """
    yaml_path = BASE_PATH / yaml_file
    assert os.path.exists(yaml_path), f"{yaml_path} does not exist."

    try:
        subprocess.check_call([sys.executable, BASE_PATH / "neps_run.py", yaml_path])
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"NePS run failed for configuration: {yaml_file} with error: {str(e)}"
        )


@pytest.mark.neps_api
def test_run_with_yaml_and_run_pipeline() -> None:
    """
    Test 'neps.run' with simple_example.yaml as run_args + a run_pipeline that is
    provided separately.
    """
    yaml_path = BASE_PATH / "simple_example.yaml"
    assert os.path.exists(yaml_path), f"{yaml_path} does not exist."

    try:
        subprocess.check_call(
            [sys.executable, BASE_PATH / "neps_run.py", yaml_path, "--evaluate_pipeline"]
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"NePS run failed for configuration: simple_example.yaml with error: {str(e)}"
        )
