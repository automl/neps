"""Samples a round of configurations, and submits a Slurm job for each.

`scalefm.acquire(..., is_async=True)` only samples: every configuration is written
to `results/async_evaluation/configs/config_<id>/config.yaml` and stays pending until
its job, `run_pipeline.py`, reports the result with `scalefm.save_pipeline_results`.
Run this again for the next round, it only submits the new configurations.
"""

import os
from pathlib import Path

import yaml

import scalefm

ROOT_DIRECTORY = Path("results/async_evaluation")

# #CHANGE_ME: a Slurm partition of your cluster (or export SBATCH_PARTITION).
SLURM_PARTITION = os.environ.get("SBATCH_PARTITION", "CHANGE_ME__PARTITION_NAME")


class ExampleSpace(scalefm.PipelineSpace):
    optimizer = scalefm.Categorical(choices=["sgd", "adam"])
    lr = scalefm.Float(lower=10e-7, upper=10e-3, log=True)


def submit(config_dir: Path) -> None:
    """Write the Slurm script of one sampled configuration and submit it."""
    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    pipeline_id = config_dir.name.removeprefix("config_")
    out_dir = Path("output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name=mnist_toy
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --output={out_dir}/%j.out
#SBATCH --error={out_dir}/%j.err

python run_pipeline.py --learning-rate {config["lr"]} \\
                       --optimizer {config["optimizer"]} \\
                       --root-directory {ROOT_DIRECTORY} \\
                       --pipeline-id {pipeline_id} \\
                       --pipeline-directory {config_dir} \\
                       --previous-pipeline-directory None \\
"""
    script_path = config_dir / "submit.sh"
    print(f"Submitting the script {script_path} (see below): \n\n{script}")
    script_path.write_text(script)
    os.system(f"sbatch {script_path}")


if __name__ == "__main__":
    if "CHANGE_ME" in SLURM_PARTITION:
        raise ValueError(
            "Set SLURM_PARTITION in this file (or export SBATCH_PARTITION) to a "
            "Slurm partition of your cluster."
        )

    scalefm.acquire(
        pipeline_space=ExampleSpace(),
        root_directory=ROOT_DIRECTORY,
        is_async=True,  # Only sample, the Slurm jobs evaluate the configurations
        worker_evaluations_to_spend=2,  # Configurations in this round
    )

    for config_dir in sorted((ROOT_DIRECTORY / "configs").glob("config_*")):
        if not (config_dir / "submit.sh").exists():  # Not submitted in an earlier round
            submit(config_dir)
