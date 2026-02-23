"""SLURM job submission manager."""

import subprocess
from pathlib import Path
from neps_cluster.slurm_config import SlurmConfig


class JobSubmitter:
    """Handles SLURM job submission."""

    def __init__(self, config: SlurmConfig):
        self.config = config
        self.last_job_id: int | None = None

    def submit(
        self,
        script: str,
        script_path: Path | None = None,
        wait: bool = False,
    ) -> int | None:
        """Submit a SLURM script.

        Args:
            script: The SLURM script content.
            script_path: Path to save the script. If None, uses a temp path.
            wait: If True, wait for job to complete before returning.

        Returns:
            Job ID if submission successful, None otherwise.
        """
        if script_path is None:
            script_path = self.config.output_dir / "submit.sh"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        script_path.chmod(0o755)

        sbatch_cmd = ["sbatch"]
        if wait:
            sbatch_cmd.append("--wait")
        sbatch_cmd.append(str(script_path))

        print(f"Submitting SLURM job: {' '.join(sbatch_cmd)}")

        result = subprocess.run(
            sbatch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return None

        stdout = result.stdout.strip()
        if stdout.startswith("Submitted batch job"):
            job_id = int(stdout.split()[-1])
            self.last_job_id = job_id
            print(f"Successfully submitted job with ID: {job_id}")
            return job_id

        print(f"Unexpected output from sbatch: {stdout}")
        return None

    def get_last_job_id(self) -> int | None:
        """Get the ID of the last submitted job."""
        return self.last_job_id
