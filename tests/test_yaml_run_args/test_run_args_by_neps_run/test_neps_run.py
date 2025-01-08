from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

BASE_PATH = Path("tests") / "test_yaml_run_args" / "test_run_args_by_neps_run"


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "config",
    [
        {"file_name": "config.yaml"},
        {"file_name": "loading_pipeline_space.yaml"},
        {"file_name": "loading_optimizer.yaml"},
        {
            "file_name": "config_select_bo.yaml",
            "check_optimizer": True,
            "optimizer_path": "select_bo_run_args.yaml",
            "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_bo/optimizer_info.yaml",  # noqa: E501
        },
        {
            "file_name": "config_priorband_with_args.yaml",
            "check_optimizer": True,
            "optimizer_path": "priorband_args_run_args.yaml",
            "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_priorband/optimizer_info.yaml",  # noqa: E501
        },
        {
            "file_name": "config_hyperband_mixed_args.yaml",
            "check_optimizer": True,
            "optimizer_path": "hyperband_searcher_kwargs_yaml_args.yaml",
            "result_path": "tests_tmpdir/test_run_args_by_neps_run/optimizer_hyperband/optimizer_info.yaml",  # noqa: E501
            "args": True,
        },
    ],
)
def test_run_with_yaml(config: dict) -> None:
    """Test "neps.run" with various run_args.yaml settings to simulate loading options
    for variables.
    """
    file_name = config["file_name"]
    check_optimizer = config.pop("check_optimizer", False)
    assert (BASE_PATH / file_name).exists(), f"{file_name} " f"does not exist."

    cmd = [
        sys.executable,
        BASE_PATH / "neps_run.py",
        BASE_PATH / file_name,
    ]
    if "args" in config:
        cmd.append("--kwargs_flag")

    try:
        subprocess.check_call(cmd)  # noqa: S603
    except subprocess.CalledProcessError:
        pytest.fail(f"NePS run failed for configuration: {file_name}")

    if check_optimizer:
        optimizer_path = config.pop("optimizer_path")
        result_path = config.pop("result_path")
        compare_generated_yaml(result_path, optimizer_path)


def compare_generated_yaml(result_path, optimizer_path):
    """Compare generated optimizer settings and solution settings."""
    assert result_path.exists(), "Generated YAML file does not exist."

    assert (
        BASE_PATH / "optimizer_yamls" / optimizer_path
    ).exists(), "Solution YAML file does not exist."

    with result_path.open("r") as gen_file:
        generated_content = yaml.safe_load(gen_file)

    with (BASE_PATH / "optimizer_yamls" / optimizer_path).open("r") as ref_file:
        reference_content = yaml.safe_load(ref_file)

    assert (
        generated_content == reference_content
    ), "The generated YAML does not match the reference YAML"
