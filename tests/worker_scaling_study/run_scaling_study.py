"""Submits one Slurm job per `neps.run` worker, for each worker count of the study.
Every worker gets its own 1-GPU allocation; the workers of a setting share a root
directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from train import NUM_WORKERS

# #CHANGE_ME: the parallelism levels to benchmark, and the sweep size they all
# share. Every count must divide TOTAL_EVALUATIONS, and TOTAL_EVALUATIONS must
# equal the grid size of `worker.HPOSpace`.
WORKER_COUNTS = (1, 2, 4, 8)
TOTAL_EVALUATIONS = 8

# #CHANGE_ME: Slurm settings for one worker. Each job is one GPU with its own
# CPUs and memory, so a worker never competes with another for the input
# pipeline -- what is being measured is parallel search, not node contention.
PARTITION = "CHANGE_ME_PARTITION_NAME"
MEM_PER_GPU = "32G"
CPUS_PER_WORKER = NUM_WORKERS + 1
TIME_LIMIT = "01:00:00"

SOURCE_DIR = Path(__file__).parent.resolve()
ROOT_DIRECTORY = SOURCE_DIR.parent / "results" / "scaling_study"


def root_dir_for(n_workers: int) -> Path:
    """Where one setting's workers share their NePS state. NePS owns this path."""
    return ROOT_DIRECTORY / f"workers_{n_workers}"


def job_dir_for(n_workers: int) -> Path:
    """Where one setting's job scripts and Slurm logs live, outside its NePS state."""
    return ROOT_DIRECTORY / "jobs" / f"workers_{n_workers}"


def write_job_script(n_workers: int, worker_index: int) -> Path:
    """The sbatch script for a single worker: one `neps.run` on one GPU."""
    job_dir = job_dir_for(n_workers)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Only the first worker of a setting creates the NePS state; the rest wait
    # for it, since `NePSState.create_or_load` does not lock its creation path.
    wait_flag = " --wait_for_state" if worker_index > 0 else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=vlm_scaling_{n_workers}w_{worker_index}
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS_PER_WORKER}
#SBATCH --mem-per-gpu={MEM_PER_GPU}
#SBATCH --time={TIME_LIMIT}
#SBATCH --chdir={SOURCE_DIR}
#SBATCH --output={job_dir}/worker_{worker_index}.out
#SBATCH --error={job_dir}/worker_{worker_index}.err

python worker.py \\
    --root_dir {root_dir_for(n_workers)} \\
    --evaluations_to_spend {TOTAL_EVALUATIONS // n_workers} \\
    --n_workers {n_workers} \\
    --worker_id worker_{worker_index}{wait_flag}
"""
    script_path = job_dir / f"worker_{worker_index}.sh"
    script_path.write_text(script)
    return script_path


def submit(script_path: Path, after_job_id: str | None = None) -> str:
    """Submit one worker job, optionally only once `after_job_id` has started."""
    command = ["sbatch"]
    if after_job_id is not None:
        command.append(f"--dependency=after:{after_job_id}")
    command.append(str(script_path))
    submission = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=True
    )
    return submission.stdout.strip().split()[-1]


def submit_setting(n_workers: int) -> list[str]:
    """Submit all of one setting's workers. The first one runs before the rest.

    The others depend on it having *started*, so no GPU sits idle in
    `worker.wait_for_state` waiting for a job that is still queued.
    """
    first_job_id = submit(write_job_script(n_workers, 0))
    return [first_job_id] + [
        submit(write_job_script(n_workers, i), after_job_id=first_job_id)
        for i in range(1, n_workers)
    ]


def main():
    for n_workers in WORKER_COUNTS:
        if TOTAL_EVALUATIONS % n_workers:
            raise ValueError(
                f"n_workers={n_workers} does not divide "
                f"TOTAL_EVALUATIONS={TOTAL_EVALUATIONS}."
            )
        submit_setting(n_workers)


if __name__ == "__main__":
    main()
