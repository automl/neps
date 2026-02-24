"""Configuration and utilities for SLURM job submission."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""

    # Job and directory settings
    root_dir: Path | str | None = None  # e.g., "results/my_experiment" or "/home/user/results/my_exp"
    config_id: str | int | None = None  # e.g., "001" or 1 (used to generate job_name if needed)
    job_name: str | None = None  # Auto-generated if None using root_dir and config_id
    
    # SLURM resource allocation
    partition: str = "gpu"
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 32
    time_hours: int = 24
    output_dir: Path | str = "slurm_logs"
    workspace_dir: Path | str = "."
    
    # Environment management
    environment_manager: Literal["conda", "virtualenv"] = "conda"
    
    # Conda settings
    conda_path: str | None = None  # e.g., "/home/user/miniconda3" (uses default ~/miniconda3 if None)
    conda_env_name: str | None = "test_scaling_laws"  # Used if environment_manager == "conda"
    
    # Virtualenv settings
    venv_path: str | None = None  # Used if environment_manager == "virtualenv"
    
    # Optional: modules to load (e.g., for HPC clusters like Juwels)
    modules_to_load: list[str] = field(default_factory=list)
    
    # Optional: SLURM account for billing
    account: str | None = None

    def __post_init__(self):
        """Validate, convert paths, and auto-generate job_name if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.workspace_dir, str):
            self.workspace_dir = Path(self.workspace_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-generate job_name if not provided
        if self.job_name is None:
            self.job_name = self._generate_job_name()
        
        # Convert root_dir to Path if provided
        if self.root_dir is not None and isinstance(self.root_dir, str):
            self.root_dir = Path(self.root_dir)
        
        # Validate environment configuration
        if self.environment_manager == "virtualenv" and not self.venv_path:
            raise ValueError(
                "venv_path must be provided when environment_manager='virtualenv'"
            )
        if self.environment_manager == "conda" and not self.conda_env_name:
            raise ValueError(
                "conda_env_name must be provided when environment_manager='conda'"
            )
    
    def _generate_job_name(self) -> str:
        """Generate a job name from root_dir and config_id.
        
        Format: {dir_prefix}_{id}
        - dir_prefix: first 4 characters of root directory name
        - id: config_id (truncated to fit within ~8 char limit)
        
        Example: If root_dir="results/scaling_study" and config_id="0042",
                 job_name will be "scal_0042"
        
        Returns:
            Generated job name (max ~8 chars for SLURM compatibility)
        """
        if self.root_dir is None or self.config_id is None:
            return "train_job"  # Default fallback
        
        # Get the directory name from root_dir
        root_path = Path(self.root_dir)
        dir_name = root_path.name.lower()
        
        # Take first 4 characters of directory name
        dir_prefix = dir_name.split("/")[-1][:4]
        
        # Convert config_id to string and take last 3 chars (or however many fit)
        config_id_str = str(self.config_id).split("_")[-1]
        
        # Build job_name: prefix_id (aiming for ~8 chars total)
        job_name = f"{dir_prefix}_{config_id_str}"
        
        return job_name


def get_environment_setup(config: SlurmConfig) -> str:
    """Get the environment setup commands based on configuration.
    
    Args:
        config: SlurmConfig object with environment settings.
        
    Returns:
        Shell commands to set up the environment.
    """
    setup_commands = []
    
    # Load modules if specified (useful for HPC clusters like Juwels)
    if config.modules_to_load:
        setup_commands.append("ml --force purge")
        for module in config.modules_to_load:
            setup_commands.append(f"ml {module}")
    
    # Activate environment
    if config.environment_manager == "conda":
        if not config.conda_env_name:
            raise ValueError("conda_env_name required for conda environment manager")
        
        # Use custom conda path if provided, otherwise default to ~/miniconda3
        conda_path = config.conda_path or "~/miniconda3"
        setup_commands.append(
            f"source {conda_path}/bin/activate && conda activate {config.conda_env_name}"
        )
    elif config.environment_manager == "virtualenv":
        if not config.venv_path:
            raise ValueError("venv_path required for virtualenv environment manager")
        setup_commands.append(f"source {config.venv_path}/bin/activate")
    else:
        raise ValueError(
            f"Unknown environment_manager: {config.environment_manager}. "
            "Must be 'conda' or 'virtualenv'"
        )
    
    return "\n".join(setup_commands)


def get_account_header(config: SlurmConfig) -> str:
    """Get the SLURM account header if configured.
    
    Args:
        config: SlurmConfig object.
        
    Returns:
        SLURM account header line (empty string if not configured).
    """
    if config.account:
        return f"#SBATCH --account={config.account}\n"
    return ""
