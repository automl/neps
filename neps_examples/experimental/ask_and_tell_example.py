"""
# AskAndTell Example: Custom Trial Execution with NePS

This script demonstrates how to use the `AskAndTell` interface from NePS to implement a custom trial execution workflow. 
The `AskAndTell` interface provides full control over the evaluation loop, allowing you to manage how trials are executed 
and results are reported back to the optimizer. This is particularly useful when you need to handle trial execution manually.

## Aim of This File

The goal of this script is to run a **successive halving** optimization process with 3 rungs. The first rung will evaluate 
9 trials in parallel. The trials are managed manually using the `AskAndTell` interface, and the SLURM scheduler is used 
to execute the trials. This setup demonstrates how to efficiently manage parallel trial execution and integrate NePS 
with external job schedulers.

## How to Use This Script

1. **Define the Search Space**:
   The search space is defined using `neps.SearchSpace`.

2. **Initialize the Optimizer**:
   We use the `successive_halving` algorithm from NePS to optimize the search space. The optimizer is wrapped with 
   the `AskAndTell` interface to enable manual control of the evaluation loop.

3. **Submit Jobs**:
   - The `submit_job` function submits a job to the SLURM scheduler using a generated script.
   - The `get_job_script` function generates a SLURM job script that executes the `train_worker` function for a given trial.

4. **Train Worker**:
   - The `train_worker` function reads the trial configuration, evaluates a dummy objective function, and writes the 
     results to a JSON file.

5. **Main Loop**:
   - The `main` function manages the optimization process:
     - It launches initial jobs based on the number of parallel trials specified.
     - It monitors the status of active jobs, retrieves results, and submits new trials as needed.
     - The loop continues until all trials are completed.

6. **Run the Script**:
   - Use the command line to run the script:
     ```bash
     python ask_and_tell_example.py --parallel 9 --results-dir results
     ```
   - `--parallel`: Specifies the number of trials to evaluate in parallel initially.
   - `--results-dir`: Specifies the directory where results will be saved.

## Key Features Demonstrated
- Custom trial execution using SLURM.
- Integration of NePS optimizers with manual control over the evaluation loop.
- Efficient management of parallel trials and result reporting.

This script serves as a template for implementing custom trial execution workflows with NePS.
"""
import argparse
import time
from pathlib import Path
import json
import neps
import os
import subprocess
import json, sys

from neps.optimizers.ask_and_tell import AskAndTell

def submit_job(pipeline_directory: Path, script: str) -> int:
    script_path = pipeline_directory / "submit.sh"
    print(f"Submitting the script {script_path} (see below): \n\n{script}")

    # You may want to remove the below check and not ask before submitting every time
    script_path.write_text(script)
    os.system(f"sbatch {script_path}")
    output = subprocess.check_output(["sbatch", str(script_path)]).decode().strip()
    job_id = int(output.split()[-1])
    return job_id

def get_job_script(pipeline_directory, trial_file):
    script = f"""#!/bin/bash
    #SBATCH --job-name=mnist_toy
    #SBATCH --partition=bosch_cpu-cascadelake
    #SBATCH --output={pipeline_directory}/%j.out
    #SBATCH --error={pipeline_directory}/%j.err
    python -c "import neps.neask_andtell_example; ask_andtell_example.train_worker('{trial_file}')"
    """
    return script

def train_worker(trial_file):
    trial_file = Path(trial_file)
    with open(trial_file) as f:
        trial = json.load(f)

    config = trial["config"]
    # Dummy objective
    loss = (config["a"] - 0.5)**2 + ((config["b"] + 2)**2) / 5

    out_file = trial_file.parent / f"result_{trial['id']}.json"
    with open(out_file, "w") as f:
        json.dump({"loss": loss}, f)

def main(parallel: int, results_dir: Path):
    class MySpace(neps.PipelineSpace):
        a = neps.Fidelity(neps.Integer(1, 13))
        b = neps.Float(1, 5)
    space = MySpace()
    opt = neps.algorithms.neps_hyperband(space, eta=3)
    ask_tell = AskAndTell(opt)

    results_dir.mkdir(exist_ok=True, parents=True)
    active = {}

    # launch initial jobs
    for _ in range(parallel):
        trial = ask_tell.ask()
        if trial is None:
            break
        trial_file = results_dir / f"trial_{trial.id}.json"
        with open(trial_file, "w") as f:
            json.dump({"id": trial.id, "config": trial.config}, f)
        job_id = submit_job(results_dir, get_job_script(results_dir, trial_file))
        active[job_id] = trial

    # monitor loop
    while active:
        for job_id, trial in list(active.items()):
            result_file = results_dir / f"result_{trial.id}.json"
            if result_file.exists():
                result = json.load(result_file.open())
                ask_tell.tell(trial, {"objective_to_minimize": result["loss"]})
                del active[job_id]
                new_trial = ask_tell.ask()
                if new_trial:
                    new_file = results_dir / f"trial_{new_trial.id}.json"
                    json.dump({"id": new_trial.id, "config": new_trial.config}, new_file.open("w"))
                    new_job_id = submit_job(results_dir, get_job_script(results_dir, new_file))
                    active[new_job_id] = new_trial
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel", type=int, default=9, 
        help="Number of trials to evaluate in parallel initially"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), 
        help="Path to save the results inside"
    )
    args = parser.parse_args()
    main(args.parallel, args.results_dir)
