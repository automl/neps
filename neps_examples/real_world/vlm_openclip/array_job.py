"""Groups pending configs by `batch_size` and writes one Slurm array job per
resource tier (see `resource_map.json`).
Run once after `generate_configs.py` has sampled some configs:

    python array_job.py
    python array_job.py --rerun_crashed   # also re-queue trials reported as crashed
    sbatch results/hpo_vlm_openclip/array_jobs/array_job_<tier>.sh
"""

import argparse
import json
from pathlib import Path

import yaml

from generate_configs import ROOT_DIRECTORY

# The training process plus its `train.NUM_WORKERS` dataloader workers (keep in
# sync with that constant), so image decoding never starves the GPU.
CPUS_PER_TASK = 4 + 1


SOURCE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = Path(ROOT_DIRECTORY).resolve()
ARRAY_JOBS_DIR = ROOT_DIR / "array_jobs"


def pending_configs(rerun_crashed=False):
    for config_dir in sorted(ROOT_DIR.glob("configs/config_*")):
        if not (config_dir / "config.yaml").exists():
            continue
        report_path = config_dir / "report.yaml"
        if not report_path.exists():
            yield config_dir
        elif rerun_crashed and yaml.safe_load(report_path.read_text()).get("reported_as") == "crashed":
            yield config_dir


def resource_tier(batch_size, tiers):
    for tier in tiers:
        if tier["max_batch_size"] is None or batch_size <= tier["max_batch_size"]:
            return tier
    raise ValueError(f"No resource tier in resource_map.json covers batch_size={batch_size}")


def write_array_job(tier, config_ids):
    ARRAY_JOBS_DIR.mkdir(parents=True, exist_ok=True)

    group_file = ARRAY_JOBS_DIR / f"array_group_{tier['name']}.yaml"
    group_file.write_text(yaml.safe_dump(config_ids))

    exclude = f"#SBATCH --exclude={','.join(EXCLUDE_NODES)}\n" if EXCLUDE_NODES else ""
    script = f"""#!/bin/bash
#SBATCH --job-name=vlm_openclip_{tier["name"]}
#SBATCH --array=0-{len(config_ids) - 1}
#SBATCH --partition={tier["partition"]}
#SBATCH --gres={tier["gres"]}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={tier["mem"]}
{exclude}#SBATCH --chdir={SOURCE_DIR}
#SBATCH --output={ARRAY_JOBS_DIR}/logs/%x-%A_%a.out
#SBATCH --error={ARRAY_JOBS_DIR}/logs/%x-%A_%a.err

mkdir -p {ARRAY_JOBS_DIR}/logs
python train.py --group_file {group_file} --task_id $SLURM_ARRAY_TASK_ID --root_dir {ROOT_DIR}
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

    for config_dir in pending_configs(rerun_crashed=args.rerun_crashed):
        config = yaml.safe_load((config_dir / "config.yaml").read_text())
        tier = resource_tier(config["batch_size"], tiers)
        groups[tier["name"]].append(config_dir.name.removeprefix("config_"))

    for tier in tiers:
        config_ids = groups[tier["name"]]
        if not config_ids:
            continue
        script_path = write_array_job(tier, config_ids)
        print(
            f"{tier['name']}: {len(config_ids)} configs on {tier['partition']}"
            f" -> sbatch {script_path}"
        )


if __name__ == "__main__":
    main()
