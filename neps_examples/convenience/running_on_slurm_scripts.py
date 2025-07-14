"""Example that shows HPO with NePS based on a slurm script."""

import logging
import os
import time
from pathlib import Path

import neps


def _submit_job(pipeline_directory: Path, script: str):
    script_path = pipeline_directory / "submit.sh"
    logging.info(f"Submitting the script {script_path} (see below): \n\n{script}")

    # You may want to remove the below check and not ask before submitting every time
    if input("Ok to submit? [Y|n] -- ").lower() in {"y", ""}:
        script_path.write_text(script)
        os.system(f"sbatch {script_path}")
    else:
        raise ValueError("We generated a slurm script that should not be submitted.")


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
python -c "import neps; neps.save_results('{pipeline_directory}', dict(objective_to_minimize=0.5, cost=1.0, learning_curve=[0.1, 0.2, 0.3]))"
"""

    return _submit_job(pipeline_directory, script)

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
