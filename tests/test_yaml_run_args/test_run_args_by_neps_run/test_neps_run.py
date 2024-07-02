import pytest
import subprocess
import os
import sys
import yaml
BASE_PATH = "tests/test_yaml_run_args/test_run_args_by_neps_run/"


@pytest.mark.neps_api
@pytest.mark.parametrize("config", [
    {"file_name": "config.yaml"},
    {"file_name": "loading_pipeline_space.yaml"},
    {"file_name": "loading_optimizer.yaml"},
    {"file_name": "config_select_bo.yaml", "check_optimizer": True, "optimizer_path":
        "select_bo_run_args.yaml",
     "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_bo"
                    "/.optimizer_info.yaml"},
    {"file_name": "config_priorband_with_args.yaml", "check_optimizer": True,
     "optimizer_path": "priorband_args_run_args.yaml",
     "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_priorband"
                    "/.optimizer_info.yaml"},
    {"file_name": "config_hyperband_mixed_args.yaml", "check_optimizer": True,
     "optimizer_path": "hyperband_searcher_kwargs_yaml_args.yaml",
     "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_hyperband"
                    "/.optimizer_info.yaml", "args": True}
])
def test_run_with_yaml(config: dict) -> None:
    """Test "neps.run" with various run_args.yaml settings to simulate loading options
    for variables."""
    file_name = config["file_name"]
    check_optimizer = config.pop("check_optimizer", False)
    assert os.path.exists(os.path.join(BASE_PATH, file_name)), (f"{file_name} "
                                                                f"does not exist.")

    cmd = [sys.executable, os.path.join(BASE_PATH, 'neps_run.py'),
           os.path.join(BASE_PATH, file_name)]
    if "args" in config:
        cmd.append('--kwargs_flag')

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        pytest.fail(f"NePS run failed for configuration: {file_name}")

    if check_optimizer:
        optimizer_path = config.pop("optimizer_path")
        result_path = config.pop("result_path")
        compare_generated_yaml(result_path, optimizer_path)


def compare_generated_yaml(result_path, optimizer_path):

    assert os.path.exists(result_path), \
        "Generated YAML file does not exist."

    assert os.path.exists(BASE_PATH + "optimizer_yamls/" + optimizer_path), \
        "Solution YAML file does not exist."

    with open(result_path, 'r') as gen_file:
        generated_content = yaml.safe_load(gen_file)

    with open(BASE_PATH + "optimizer_yamls/" + optimizer_path, 'r') as ref_file:
        reference_content = yaml.safe_load(ref_file)

    assert generated_content == reference_content, \
        "The generated YAML does not match the reference YAML"
