from pathlib import Path
import subprocess
import neps
import os

def _submit_job(pipeline_directory: Path, script: str):
    script_path = pipeline_directory / "submit.sh"
    print(f"Submitting the script {script_path} (see below): \n\n{script}")

    # You may want to remove the below check and not ask before submitting every time
    script_path.write_text(script)
    os.system(f"sbatch {script_path}")

def evaluate_pipeline_via_slurm(pipeline_id, pipeline_directory, lr, optimizer):
    print(f"{pipeline_id} optimization_dir")
    out_dir = Path('output_dir')
    out_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name=mnist_toy
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --output={out_dir}/%j.out
#SBATCH --error={out_dir}/%j.err

python run_pipeline.py --learning_rate {lr} \\
                       --optimizer {optimizer} \\
                       --root_directory {"results"} \\
                       --pipeline_id {pipeline_id}
"""

    return _submit_job(pipeline_directory, script)


pipeline_space = dict(
    optimizer=neps.Categorical(choices=["sgd", "adam"]),
    lr=neps.Float(lower=10e-7, upper=10e-3, log=True),
)

neps.run(
    evaluate_pipeline=evaluate_pipeline_via_slurm,
    pipeline_space=pipeline_space,
    root_directory="results",
    max_evaluations_total=3,
    post_run_summary=True,
)
