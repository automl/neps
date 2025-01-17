"""Example that shows HPO with NePS based on a slurm script."""

import logging
import os
import time
from pathlib import Path

import neps


def _ask_to_submit_slurm_script(pipeline_directory: Path, script: str):
    script_path = pipeline_directory / "submit.sh"
    logging.info(f"Submitting the script {script_path} (see below): \n\n{script}")

    # You may want to remove the below check and not ask before submitting every time
    if input("Ok to submit? [Y|n] -- ").lower() in {"y", ""}:
        script_path.write_text(script)
        os.system(f"sbatch {script_path}")
    else:
        raise ValueError("We generated a slurm script that should not be submitted.")


def _get_validation_error(pipeline_directory: Path):
    validation_error_file = pipeline_directory / "validation_error_from_slurm_job.txt"
    if validation_error_file.exists():
        return float(validation_error_file.read_text())
    return None


def evaluate_pipeline_via_slurm(
    pipeline_directory: Path, optimizer: str, learning_rate: float
):
    script = f"""#!/bin/bash
#SBATCH --time 0-00:05
#SBATCH --job-name test
#SBATCH --partition cpu-cascadelake
#SBATCH --error "{pipeline_directory}/%N_%A_%x_%a.oe"
#SBATCH --output "{pipeline_directory}/%N_%A_%x_%a.oe"
# Plugin your python script here
python -c "print('Learning rate {learning_rate} and optimizer {optimizer}')"
# At the end of training and validation create this file
echo -10 > {pipeline_directory}/validation_error_from_slurm_job.txt
"""

    # Now we submit and wait until the job has created validation_error_from_slurm_job.txt
    _ask_to_submit_slurm_script(pipeline_directory, script)
    while validation_error := _get_validation_error(pipeline_directory) is None:
        logging.info("Waiting until the job has finished.")
        time.sleep(60)  # Adjust to something reasonable
    return validation_error


pipeline_space = dict(
    optimizer=neps.Categorical(choices=["sgd", "adam"]),
    learning_rate=neps.Float(lower=10e-7, upper=10e-3, log=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline_via_slurm,
    pipeline_space=pipeline_space,
    root_directory="results/slurm_script_example",
    max_evaluations_total=5,
)
