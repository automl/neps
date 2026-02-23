"""Script builders for various SLURM job types."""

from pathlib import Path
from neps_cluster.slurm_config import SlurmConfig, get_environment_setup, get_account_header


class BasicScriptBuilder:
    """Builds a basic SLURM script for single job execution."""

    def __init__(self, config: SlurmConfig):
        self.config = config

    def build(self, cmd: str, job_name: str | None = None) -> str:
        """Build a basic SLURM script.

        Args:
            cmd: The command to execute.
            job_name: Override job name for this specific script.

        Returns:
            The complete SLURM script as a string.
        """
        job_name = job_name or self.config.job_name
        env_setup = get_environment_setup(self.config)
        account = get_account_header(self.config)

        script = f"""#!/bin/bash
#SBATCH --nodes={self.config.nodes}
#SBATCH --gres=gpu:{self.config.gpus_per_node}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --job-name={job_name}
#SBATCH --time={self.config.time_hours}:00:00
{account}#SBATCH -p {self.config.partition}
#SBATCH --output {self.config.output_dir}/%x-%j.out
#SBATCH --error {self.config.output_dir}/%x-%j.err

{env_setup}

echo "Started at $(date)"
start=$(date +%s)

cd {self.config.workspace_dir}

{cmd}

end=$(date +%s)
runtime=$((end - start))
echo "Completed at $(date), runtime: ${{runtime}}s"
"""
        return script


class ArrayScriptBuilder:
    """Builds an array SLURM script for parallel job execution."""

    def __init__(self, config: SlurmConfig):
        self.config = config

    def build(
        self, cmd_template: str, array_size: int, job_name: str | None = None
    ) -> str:
        """Build an array SLURM script.

        Args:
            cmd_template: Command template where {index} will be replaced with array index.
            array_size: Number of array jobs.
            job_name: Override job name for this specific script.

        Returns:
            The complete SLURM array script as a string.
        """
        job_name = job_name or self.config.job_name
        env_setup = get_environment_setup(self.config)
        account = get_account_header(self.config)

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:{self.config.gpus_per_node}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --job-name={job_name}
#SBATCH --time={self.config.time_hours}:00:00
#SBATCH --array=1-{array_size}
{account}#SBATCH -p {self.config.partition}
#SBATCH --output {self.config.output_dir}/%x-%A_%a.out
#SBATCH --error {self.config.output_dir}/%x-%A_%a.err

{env_setup}

echo "Started at $(date)"
echo "Array task: $SLURM_ARRAY_TASK_ID"

cd {self.config.workspace_dir}

{cmd_template}

echo "Completed at $(date)"
"""
        return script
