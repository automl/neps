"""Async evaluation wrapper for NEPS integration with SLURM."""

from pathlib import Path
from cluster.slurm_config import SlurmConfig
from cluster.slurm_script_builders import BasicScriptBuilder
from cluster.job_submitter import JobSubmitter


class NepsAsyncEvaluator:
    def __init__(
        self,
        training_script: str | Path,
        config: SlurmConfig | None = None,
    ):
        self.training_script = Path(training_script)
        self.config = config or SlurmConfig()
        self.submitter = JobSubmitter(self.config)
        self.script_builder = BasicScriptBuilder(self.config)

    def lazy_submit(
        self,
        pipeline_id: str,
        **kwargs,
    ) -> None:
        args = self._build_args(pipeline_id=pipeline_id, **kwargs)

        cmd = f"python {self.training_script} {args}"

        script_builder = self.script_builder

        script = script_builder.build(cmd=cmd)

        script_path = self.config.output_dir / f"{pipeline_id}_submit.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        script_path.chmod(0o755)
        # self.submitter.submit(script=script, script_path=script_path)

    @staticmethod
    def _build_args(pipeline_id: str, **kwargs) -> str:
        args = [f"--pipeline-id {pipeline_id}"]

        for key, value in kwargs.items():
            arg_name = f"--{key.replace('_', '-')}" #TODO: convention for arg names?

            if isinstance(value, bool):
                if value:
                    args.append(arg_name)
            else:
                args.append(f'{arg_name} "{value}"')

        return " ".join(args)
