"""Groups pending configs by `recommended_num_gpus` and writes one Slurm array job
per GPU count (see `resource_map.json`), as scan_llm_bench's
`cluster/slurm_manager.py train_array_grouped` does.
Run once after `prepare_config_resources.py` has prepared the sampled configs:

    python array_job.py
    python array_job.py --rerun_crashed   # also re-queue trials reported as crashed
    sbatch results/hpo_llm_pretraining/array_jobs/array_job_<tier>.sh
"""


import argparse
import json
from pathlib import Path

import yaml
from generate_configs import ROOT_DIRECTORY

# CPUs for each of a job's training processes (one per GPU).
CPUS_PER_GPU = 4

# #CHANGE_ME: nodes to keep jobs off (e.g. ones with a faulty GPU), or leave empty.
EXCLUDE_NODES: list[str] = []


SOURCE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = Path(ROOT_DIRECTORY).resolve()
ARRAY_JOBS_DIR = ROOT_DIR / "array_jobs"


def pending_configs(*, rerun_crashed=False):
    for config_dir in sorted(ROOT_DIR.glob("configs/config_*")):
        if not (config_dir / "config.yaml").exists():
            continue
        report_path = config_dir / "report.yaml"
        if not report_path.exists() or (
            rerun_crashed
            and yaml.safe_load(report_path.read_text()).get("reported_as") == "crashed"
        ):
            yield config_dir

def resource_tier(num_gpus, tiers):
    for tier in tiers:
        if tier["num_gpus"] == num_gpus:
            return tier
    raise ValueError(f"No entry in resource_map.json has num_gpus={num_gpus}")


def run_cmd(num_gpus):
    """The training command: python on one GPU, torchrun with one process per GPU on more."""
    cmd = f"pipeline/train.py --config_dir {ROOT_DIR}/configs/config_$CONFIG_ID --root_dir {ROOT_DIR}"
    if num_gpus > 1:
        return f"torchrun --standalone --nnodes=1 --nproc_per_node={num_gpus} {cmd}"
    return f"python {cmd}"


def write_array_job(tier, config_ids):
    # Slurm opens the --output/--error files before the script runs, so the log
    # directory has to exist at submit time.
    (ARRAY_JOBS_DIR / "logs").mkdir(parents=True, exist_ok=True)

    exclude = f"#SBATCH --exclude={','.join(EXCLUDE_NODES)}\n" if EXCLUDE_NODES else ""
    script = f"""#!/bin/bash
#SBATCH --job-name=llm_pretraining_{tier["name"]}
#SBATCH --array=0-{len(config_ids) - 1}
#SBATCH --partition={tier["partition"]}
#SBATCH --gres={tier["gres"]}
#SBATCH --cpus-per-task={CPUS_PER_GPU * tier["num_gpus"]}
#SBATCH --mem={tier["mem"]}
#SBATCH --time={tier["time"]}
{exclude}#SBATCH --chdir={SOURCE_DIR}
#SBATCH --output={ARRAY_JOBS_DIR}/logs/%x-%A_%a.out
#SBATCH --error={ARRAY_JOBS_DIR}/logs/%x-%A_%a.err

CONFIG_IDS=({" ".join(config_ids)})
CONFIG_ID=${{CONFIG_IDS[$SLURM_ARRAY_TASK_ID]}}
{run_cmd(tier["num_gpus"])}
"""
    script_path = ARRAY_JOBS_DIR / f"array_job_{tier['name']}.sh"
    script_path.write_text(script)
    return script_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun_crashed",
        action="store_true",
        help="Also queue configs whose report says they crashed (e.g. on a faulty GPU).",
    )
    args = parser.parse_args()

    tiers = json.loads((SOURCE_DIR / "resource_map.json").read_text())

    unset = [tier["name"] for tier in tiers if "CHANGE_ME" in tier["partition"]]
    if unset:
        raise ValueError(
            f"resource_map.json has placeholder partitions for tier(s) {unset}. "
            "Edit resource_map.json and set each 'partition' to a real Slurm "
            "partition name for your cluster before running array_job.py."
        )
            
    groups = {tier["name"]: [] for tier in tiers}
    unprepared = []

    for config_dir in pending_configs(rerun_crashed=args.rerun_crashed):
        config = yaml.safe_load((config_dir / "config.yaml").read_text())
        config_id = config_dir.name.removeprefix("config_")
        if "recommended_num_gpus" not in config:
            unprepared.append(config_id)
            continue
        tier = resource_tier(config["recommended_num_gpus"], tiers)
        groups[tier["name"]].append(config_id)

    for old_script in ARRAY_JOBS_DIR.glob("array_job_*.sh"):
        old_script.unlink()

    for tier in tiers:
        config_ids = groups[tier["name"]]
        if not config_ids:
            continue
        script_path = write_array_job(tier, config_ids)
        print(
            f"{tier['name']}: {len(config_ids)} configs on {tier['partition']}"
            f" -> sbatch {script_path}"
        )
        
    if unprepared:
        print(
            f"Not queued, no recommended_num_gpus yet (run prepare_config_resources.py): "
            f"configs {', '.join(unprepared)}"
        )


if __name__ == "__main__":
    main()
