"""Async evaluation wrapper for NEPS integration with SLURM."""

from pathlib import Path
from neps_cluster.slurm_config import SlurmConfig
from neps_cluster.slurm_script_builders import BasicScriptBuilder
from neps_cluster.job_submitter import JobSubmitter


class NepsAsyncEvaluator:
    """Wraps a training script for async evaluation via SLURM.

    This class is designed to work with neps.run() where the evaluate_pipeline
    function receives pipeline arguments and submits them to SLURM for execution.

    Usage:
        ```python
        evaluator = NepsAsyncEvaluator(
            training_script="train.py",
            config=SlurmConfig(partition="gpu")
        )

        def evaluate_pipeline(pipeline_id, **kwargs):
            return evaluator.submit(pipeline_id=pipeline_id, **kwargs)

        neps.run(
            evaluate_pipeline=evaluate_pipeline,
            pipeline_space=MySpace(),
            ...
        )
        ```
    """

    def __init__(
        self,
        training_script: str | Path,
        config: SlurmConfig | None = None,
    ):
        """Initialize the async evaluator.

        Args:
            training_script: Path to the training script to execute.
            config: SLURM configuration. Uses defaults if not provided.
        """
        self.training_script = Path(training_script)
        self.config = config or SlurmConfig()
        self.submitter = JobSubmitter(self.config)
        self.script_builder = BasicScriptBuilder(self.config)

    def submit(
        self,
        pipeline_id: str,
        **kwargs,
    ) -> None:
        """Build and submit a SLURM job with the given pipeline arguments.

        Args:
            pipeline_id: Unique identifier for this pipeline run (used as config_id for job naming).
            **kwargs: Additional arguments to pass to the training script.

        Returns:
            None (follows neps async evaluation pattern).
        """
        # Build command line arguments from kwargs
        args = self._build_args(pipeline_id=pipeline_id, **kwargs)

        # Build the command
        cmd = f"python {self.training_script} {args}"

        # Update config with current pipeline_id for job name generation
        # This allows job_name to be auto-generated if root_dir is set
        if self.config.job_name is None and self.config.root_dir is not None:
            # Create a temporary config with this pipeline_id for job name generation
            temp_config = SlurmConfig(
                root_dir=self.config.root_dir,
                config_id=pipeline_id,
                partition=self.config.partition,
                nodes=self.config.nodes,
                gpus_per_node=self.config.gpus_per_node,
                cpus_per_task=self.config.cpus_per_task,
                mem_gb=self.config.mem_gb,
                time_hours=self.config.time_hours,
                output_dir=self.config.output_dir,
                workspace_dir=self.config.workspace_dir,
                environment_manager=self.config.environment_manager,
                conda_path=self.config.conda_path,
                conda_env_name=self.config.conda_env_name,
                venv_path=self.config.venv_path,
                modules_to_load=self.config.modules_to_load,
                account=self.config.account,
            )
            script_builder = BasicScriptBuilder(temp_config)
        else:
            script_builder = self.script_builder

        # Build and submit the SLURM script
        script = script_builder.build(cmd=cmd)

        script_path = self.config.output_dir / f"{pipeline_id}_submit.sh"
        self.submitter.submit(script=script, script_path=script_path)

    @staticmethod
    def _build_args(pipeline_id: str, **kwargs) -> str:
        """Build command line arguments from kwargs.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional arguments.

        Returns:
            Space-separated list of command line arguments.
        """
        args = [f"--pipeline-id {pipeline_id}"]

        for key, value in kwargs.items():
            # Convert snake_case to --kebab-case
            arg_name = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    args.append(arg_name)
            else:
                args.append(f'{arg_name} "{value}"')

        return " ".join(args)
