"""Launches the GPU-throughput scaling study: one Slurm job per GPU count.

    python run_scaling_study.py

Each job runs its own independent NePS sweep (`hpo_ddp.py`) under `torchrun`
with that many GPUs, into its own `root_directory`:

    results/scaling_study/gpus_1/
    results/scaling_study/gpus_2/
    results/scaling_study/gpus_4/

The jobs are submitted and this script exits -- they queue and run
concurrently, since none of them shares an allocation with another. Every
sweep evaluates the *same* configs in the same order (see `hpo_ddp.HPOSpace`,
which is deliberately a grid), so the only difference between the three
directories is how many GPUs the identical work was spread over. That is what
makes `visualization.py` able to put them on one axis.

Once they have finished:

    python visualization.py
"""

import subprocess
from pathlib import Path

from hpo_ddp import EVALUATIONS_TO_SPEND
from train_ddp import NUM_WORKERS

# #CHANGE_ME: the GPU counts to benchmark, from the minimum to the maximum.
# Every `batch_size` choice in `hpo_ddp.HPOSpace` must divide by each entry.
N_GPUS_CHOICES = (1, 2, 4)

# #CHANGE_ME: the GPU count you actually expect to run production training on.
MAIN_LOAD_N_GPUS = 4

# #CHANGE_ME: Slurm settings. CPUs are sized per rank (the training process
# plus its dataloader workers) so image decoding never starves the GPUs -- if
# the 4-GPU job got the same CPU allocation as the 1-GPU job, it would be the
# input pipeline being measured, not the GPUs.
PARTITION = "testdlc2_gpu-h200"
MEM_PER_GPU = "32G"
CPUS_PER_RANK = NUM_WORKERS + 1
TIME_LIMIT = "01:00:00"

SOURCE_DIR = Path(__file__).parent.resolve()
ROOT_DIRECTORY = SOURCE_DIR.parent / "results" / "scaling_study"


def root_dir_for(n_gpus: int) -> Path:
    """Where the sweep for `n_gpus` keeps its NePS state.

    Nothing may create or write into this directory but NePS itself:
    `NePSState.create_or_load` treats an existing path as an existing state and
    goes looking for an `optimizer_info.yaml` that was never written. Hence the
    separate `job_dir_for` below for the job script and its logs.
    """
    return ROOT_DIRECTORY / f"gpus_{n_gpus}"


def job_dir_for(n_gpus: int) -> Path:
    """Where one sweep's job script and Slurm logs live -- outside its NePS state."""
    return ROOT_DIRECTORY / "jobs" / f"gpus_{n_gpus}"


def write_job_script(n_gpus: int) -> Path:
    """Write the sbatch script for one GPU count's sweep."""
    root_dir = root_dir_for(n_gpus)
    job_dir = job_dir_for(n_gpus)
    job_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name=vlm_scaling_{n_gpus}gpu
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_gpus * CPUS_PER_RANK}
#SBATCH --mem-per-gpu={MEM_PER_GPU}
#SBATCH --time={TIME_LIMIT}
#SBATCH --chdir={SOURCE_DIR}
#SBATCH --output={job_dir}/slurm.out
#SBATCH --error={job_dir}/slurm.err

torchrun --standalone --nproc_per_node={n_gpus} hpo_ddp.py \\
    --root_dir {root_dir} --n_gpus {n_gpus}
"""
    script_path = job_dir / "job.sh"
    script_path.write_text(script)
    return script_path


def main():
    for n_gpus in N_GPUS_CHOICES:
        script_path = write_job_script(n_gpus)
        submission = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True,
        )
        print(
            f"{n_gpus} GPU(s): {EVALUATIONS_TO_SPEND} evaluations -> "
            f"{submission.stdout.strip()} ({root_dir_for(n_gpus)})"
        )

    print("\nOnce the jobs finish: python visualization.py")


if __name__ == "__main__":
    main()
