import math
import subprocess
from pathlib import Path


class SlurmManager:
    def __init__(
        self,
        job_name: str = "OpenClip_train",
        partition: str = "gpu",
        nodes: int = 1,
        gpus_per_node: int = 4,
        cpus_per_task: int = 8,
        mem_gb: int = 32,
        root_dir: Path | str = ".",
        workspace_dir: str = ".",
        command_dir: Path | str | None = None,
        cluster: str = "meta", # not important
        venv_path: str | None = None
    ) -> None:
        """
        Manager to create and submit Slurm batch jobs.

        Args:
            job_name: Name of the Slurm job.
            partition: Slurm partition to use.
            nodes: Number of nodes.
            gpus_per_node: Number of GPUs per node.
            cpus_per_task: CPU cores per task.
            mem_gb: Memory in gigabytes.
            root_dir: root dir.
            workspace_dir: Workspace directory (where to cd).
            command_dir: Command directory (for PYTHONPATH, where code is).
        """
        self.job_id = None
        self.job_name = job_name
        self.partition = partition
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.cpus_per_task = cpus_per_task
        self.mem_gb = mem_gb
        self.script_path = Path(root_dir) / "train.sh"
        self.workspace_dir = workspace_dir
        # If command_dir not provided, use workspace_dir (code in workspace)
        self.command_dir = command_dir if command_dir else workspace_dir
        self.cluster = cluster
        self.venv_path = venv_path

    def build_script(
        self,
        cmd: str,
        save_to: Path = None,
        wandb_project_name: str = None,
    ) -> None:
        """
        Build a Slurm script that runs the given command.

        Args:
            cmd: Command to run.

        Returns:
            Path to the created script.
        """
        cuda_visible = ",".join(str(i) for i in range(self.gpus_per_node))

        if not cmd.endswith("\\"):
            cmd += " \\"

        cmd += f"\n    --workers {self.cpus_per_task}"

        ws_python_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        script = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --wait-all-nodes=1
#SBATCH --job-name={self.job_name}
{account}#SBATCH -p {self.partition}
#SBATCH --time=24:00:00

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

{self.load_env()}
{self._nccl_env()}

echo "Started at $(date)";

start=`date +%s`

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES={cuda_visible}
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

# first node to be used as the master node in multi-node distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

{ws_python_paths}

export WANDB_DIR=${{TRAIN_DIR}}/{wandb_project_name}

{cmd}

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def load_env(self) -> str:
        if self.cluster == "juwels":
            venv_activate = ""
            if self.venv_path:
                venv_activate = f"source {self.venv_path}\n"
            else:
                raise ValueError("venv_path must be provided for juwels cluster")
            return f"""\
ml --force purge
ml Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
{venv_activate}"""

        if self.cluster == "meta":
            return "source ~/miniconda3/bin/activate && conda activate test_scaling_laws"

        raise ValueError(f"Unknown cluster: {self.cluster}")

    def submit_job(self, cmd: str):
        """
        Submit the Slurm script using sbatch.
        If wait=True, block until the job finishes.
        """

        self.build_script(cmd=cmd)
        self.submit_without_build()

    def submit_without_build(self, script_path: Path = None, wait: bool = True):
        if not self.script_path.exists() and not script_path.exists():
            raise FileNotFoundError("Slurm script not found.")

        sbatch_cmd = ["sbatch"]
        if wait:
            sbatch_cmd.append("--wait")

        sbatch_cmd.append(
            str(self.script_path) if script_path is None else str(script_path))

        print("Submitting:", " ".join(sbatch_cmd))

        result = subprocess.run(
            sbatch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout = result.stdout.strip()
        job_id = None

        if stdout.startswith("Submitted batch job"):
            job_id = stdout.split()[-1]
            job_id = int(job_id)

        self.job_id = job_id
        print("Job ID:", job_id)

    def build_evaluation_script(self, cmd: str, job_name,
                                save_to: Path = None):
        ws_python_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --job-name={job_name}
{account}#SBATCH -p {self.partition}
#SBATCH --time=24:00:00

#SBATCH --output evaluator_logs/%x-%A.out
#SBATCH --error evaluator_logs/%x-%A.err

{self.load_env()}

echo "Started at $(date)";

{ws_python_paths}

{cmd}

echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_downstream_evaluation_script(self, job_name, root, model, seed, config_id,
                                           cluster, save_to: Path = None):
        if "train/" in root:
            root = root.replace("train/", "")
        python_ws_paths = self._slurm_ws_python_path_setup()
        account = self._account()
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={self.cpus_per_task}
{account}#SBATCH -p {self.partition}

#SBATCH --output=downstream_evaluator_logs/%x-%j.out
#SBATCH --error=downstream_evaluator_logs/%x-%j.err

{self.load_env()}

{python_ws_paths}

python post_hoc_iterator.py --root {root} --model {model} --seed {seed} --pipeline_id {config_id} --cluster {cluster}
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_job_for_evaluation(self, cmd: str, models_cfg_dir, nr_total_configs,
                                       save_to: Path = None,
                                       bundle: int = 4):

        array_size = math.ceil(nr_total_configs / bundle)

        cmd = cmd.strip()
        python_ws_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        step_jobs_logs_dir = Path(self.workspace_dir) / "train" / Path(save_to).parent
        step_jobs_logs_dir = str(step_jobs_logs_dir)

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task={self.cpus_per_task}
{account}#SBATCH -p {self.partition}
#SBATCH --time=24:00:00

#SBATCH --array=1-{array_size}

#SBATCH --job-name=eval_config
#SBATCH --output=evaluator_logs/%x-%A_%a.out
#SBATCH --error=evaluator_logs/%x-%A_%a.err

{self.load_env()}
i
echo "Started at $(date)"
echo "Array task id: ${{SLURM_ARRAY_TASK_ID}}"

{python_ws_paths}

LOG_DIR="{step_jobs_logs_dir}"
mkdir -p "${{LOG_DIR}}"

MODELS_CFG_DIR="{models_cfg_dir}"

BASE_ID=$(( (SLURM_ARRAY_TASK_ID - 1) * {bundle} + 1 ))

echo "Array task ${{SLURM_ARRAY_TASK_ID}} uses config ids: ${{BASE_ID}}..$((BASE_ID+{bundle}-1))"

run_one () {{
  local step_idx="$1"
  local config_ID="$2"

  if [ "${{config_ID}}" -gt "{nr_total_configs}" ]; then
        echo "Skipping step ${{step_idx}} because config_ID=${{config_ID}} exceeds nr_total_configs={nr_total_configs}"
        return 0
      fi

  RUN_NAME="train_config${{config_ID}}"
  MODEL_NAME="local-dir:${{MODELS_CFG_DIR}}/config_${{config_ID}}"

  echo "Starting eval step ${{step_idx}} (config_ID=${{config_ID}}) at $(date)"

  srun --exclusive --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \\
    --output "${{LOG_DIR}}/array_eval_logs/step_${{step_idx}}_%x-%A_%a.out" \\
    --error  "${{LOG_DIR}}/array_eval_logs/step_${{step_idx}}_%x-%A_%a.err" \\
    {cmd} \\
    --model "${{MODEL_NAME}}" \\
    --name "${{RUN_NAME}}" \\
    --pipeline-id "${{config_ID}}" &
}}

for step_idx in 0 1 2 3; do
  config_ID=$(( BASE_ID + step_idx ))
  run_one "${{step_idx}}" "${{config_ID}}"
done

wait

echo "Ended at $(date)"
echo "Job execution complete."
"""

        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_job_for_downstream_evaluation(
        self,
        root: str,
        models_cfg_dir: str,
        nr_total_configs: int,
        cluster: str,
        seed: int,
        save_to: Path = None,
        bundle: int = 4,
    ):
        array_size = math.ceil(nr_total_configs / bundle)

        python_ws_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        step_jobs_logs_dir = Path(self.workspace_dir) / "train" / Path(save_to).parent
        step_jobs_logs_dir = str(step_jobs_logs_dir)

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task={self.cpus_per_task}
{account}#SBATCH -p {self.partition}
#SBATCH --time=24:00:00

#SBATCH --array=1-{array_size}

#SBATCH --job-name=downstream_eval
#SBATCH --output=downstream_evaluator_logs/%x-%A_%a.out
#SBATCH --error=downstream_evaluator_logs/%x-%A_%a.err

{self.load_env()}

echo "Started at $(date)"
echo "Array task id: ${{SLURM_ARRAY_TASK_ID}}"

{python_ws_paths}

LOG_DIR="{step_jobs_logs_dir}"
mkdir -p "${{LOG_DIR}}"
mkdir -p "${{LOG_DIR}}/downstream_array_eval_logs"

MODELS_CFG_DIR="{models_cfg_dir}"
ROOT_DIR="{root}"
CLUSTER="{cluster}"
SEED="{seed}"

BASE_ID=$(( (SLURM_ARRAY_TASK_ID - 1) * {bundle} + 1 ))

echo "Array task ${{SLURM_ARRAY_TASK_ID}} uses config ids: ${{BASE_ID}}..$((BASE_ID+{bundle}-1))"

run_one () {{
    local step_idx="$1"
    local config_ID="$2"

    if [ "${{config_ID}}" -gt "{nr_total_configs}" ]; then
      echo "Skipping step ${{step_idx}} because config_ID=${{config_ID}} exceeds nr_total_configs={nr_total_configs}"
      return 0
    fi

    MODEL_NAME="local-dir:${{MODELS_CFG_DIR}}/config_${{config_ID}}"

    echo "Starting downstream eval step ${{step_idx}} (config_ID=${{config_ID}}) at $(date)"

    srun --exclusive --ntasks=1 --cpus-per-task={self.cpus_per_task} --gres=gpu:1 \\
      --output "${{LOG_DIR}}/downstream_array_eval_logs/step_${{step_idx}}_%x-%A_%a.out" \\
      --error  "${{LOG_DIR}}/downstream_array_eval_logs/step_${{step_idx}}_%x-%A_%a.err" \\
      python -m post_hoc_iterator \\
        --root "${{ROOT_DIR}}/train_config${{config_ID}}" \\
        --model "${{MODEL_NAME}}" \\
        --seed "${{SEED}}" \\
        --pipeline_id "${{config_ID}}" \\
        --cluster "${{CLUSTER}}" &
}}

for step_idx in 0 1 2 3; do
    config_ID=$(( BASE_ID + step_idx ))
    run_one "${{step_idx}}" "${{config_ID}}"
done

wait

echo "Ended at $(date)"
echo "Job execution complete."
"""

        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_script(self,
                           model_root: str,
                           logs_root: str,
                           config_txt: str,
                           array_size: int,
                           save_to: Path,
                           batch_size: int,
                           seed: int,
                           wandb_project_name: str,
                           ) -> None:

        cuda_visible = ",".join(str(i) for i in range(self.gpus_per_node))

        train_data = ""
        python_ws_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        script = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --wait-all-nodes=1
#SBATCH --job-name={self.job_name}
#SBATCH -p {self.partition}
#SBATCH --time=24:00:00
{account}#SBATCH --array=1-{array_size}

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error array_job_logs/%x-%A_%a.err

{self.load_env()}

{self._nccl_env()}
echo "Started at $(date)";
start=`date +%s`

export CUDA_VISIBLE_DEVICES={cuda_visible}
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

{python_ws_paths}
export WANDB_DIR=${{TRAIN_DIR}}/{wandb_project_name}

CONFIG_TXT="{config_txt}"

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "${{CONFIG_TXT}}")
[ -z "$LINE" ] && echo "No line for task $SLURM_ARRAY_TASK_ID" && exit 1

read -r config_ID TRAIN_NUM_SAMPLES LR WD WARMUP BETA1 BETA2 EPS EPOCHS <<< "$LINE"

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: EPOCHS is not an integer: $EPOCHS"
    exit 1
fi

EPOCHS=$((EPOCHS))

TRAIN_NUM_SAMPLES=${{TRAIN_NUM_SAMPLES%.*}}

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \\
    --train-data "{train_data}" \\
    --local-loss \\
    --gather-with-grad \\
    --dataset-type webdataset \\
    --precision amp \\
    --report-to wandb \\
    --wandb-project-name '{wandb_project_name}' \\
    --model "local-dir:{model_root}/model_cfg/config_${{config_ID}}" \\
    --train-num-samples "${{TRAIN_NUM_SAMPLES}}" \\
    --epochs ${{EPOCHS}} \\
    --name "train_config${{config_ID}}" \\
    --logs "{logs_root}" \\
    --seed {seed} \\
    --batch-size {batch_size} \\
    --workers {self.cpus_per_task} \\
    --pipeline-id "${{config_ID}}" \\
    --lr "${{LR}}" \\
    --wd "${{WD}}" \\
    --warmup-fraction "${{WARMUP}}" \\
    --beta1 "${{BETA1}}" \\
    --beta2 "${{BETA2}}" \\
    --eps "${{EPS}}"

end=`date +%s`
echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_script_for_single_gpu_runs(self, array_size, seed, batch_size,
                                               config_txt, logs_root,
                                               wandb_project_name, save_to,
                                               model_root=None):
        if model_root is None:
            model_root = "results/optimizer=experimental/seed=0"

        train_data = ""

        step_jobs_logs_dir = Path(self.workspace_dir) / "train" / Path(save_to).parent
        step_jobs_logs_dir = str(step_jobs_logs_dir)

        python_ws_paths = self._slurm_ws_python_path_setup()
        account = self._account()

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --job-name={self.job_name}
#SBATCH -p {self.partition}
#SBATCH --time=24:00:00
{account}#SBATCH --array=1-{array_size}

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error  array_job_logs/%x-%A_%a.err

{self.load_env()}

echo "Started at $(date)";
start=`date +%s`

{python_ws_paths}

LOG_DIR="{step_jobs_logs_dir}"
mkdir -p "${{LOG_DIR}}"
export WANDB_DIR=${{TRAIN_DIR}}/{wandb_project_name}

CONFIG_TXT="{config_txt}"
base_line=$(( (SLURM_ARRAY_TASK_ID - 1) * 4 + 1 ))

echo "Array task ${{SLURM_ARRAY_TASK_ID}}"
echo "  Using config lines: ${{base_line}}..$((base_line+3))"

run_one () {{
  local step_idx="$1"
  local line="$2"

  read -r config_ID TRAIN_NUM_SAMPLES LR WD WARMUP BETA1 BETA2 EPS EPOCHS <<< "${{line}}"
  TRAIN_NUM_SAMPLES=${{TRAIN_NUM_SAMPLES%.*}}

  if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
        echo "Error: EPOCHS is not an integer: $EPOCHS"
        exit 1
  fi

  EPOCHS=$((EPOCHS))

  echo "About to start srun for step ${{step_idx}} (config_ID=${{config_ID}}) at $(date)"

  srun --exclusive --ntasks=1 --cpus-per-task={self.cpus_per_task} --gres=gpu:1 \\
      --output "${{LOG_DIR}}/array_job_logs/step_${{step_idx}}_%x-%A_%a.out" \\
      --error  "${{LOG_DIR}}/array_job_logs/step_${{step_idx}}_%x-%A_%a.err" \\
      python -u -m clip_train.main \\
        --train-data '{train_data}' \\
        --local-loss \\
        --gather-with-grad \\
        --dataset-type webdataset \\
        --precision amp \\
        --report-to wandb \\
        --wandb-project-name '{wandb_project_name}' \\
        --model "local-dir:{model_root}/model_cfg/config_${{config_ID}}" \\
        --train-num-samples "${{TRAIN_NUM_SAMPLES}}" \\
        --epochs ${{EPOCHS}} \\
        --name "train_config${{config_ID}}" \\
        --logs "{logs_root}" \\
        --seed {seed} \\
        --batch-size {batch_size} \\
        --workers {self.cpus_per_task} \\
        --pipeline-id "${{config_ID}}" \\
        --lr "${{LR}}" \\
        --wd "${{WD}}" \\
        --warmup-fraction "${{WARMUP}}" \\
        --beta1 "${{BETA1}}" \\
        --beta2 "${{BETA2}}" \\
        --eps "${{EPS}}" &
    }}

for step_idx in 0 1 2 3; do
  line_no=$((base_line + step_idx))
  line=$(sed -n "${{line_no}}p" "${{CONFIG_TXT}}")

  if [ -z "${{line}}" ]; then
    echo "No line ${{line_no}} for array task ${{SLURM_ARRAY_TASK_ID}}, skipping"
    continue
  fi

  echo "  Line${{step_idx}}: ${{line}}"
  run_one "${{step_idx}}" "${{line}}"
done

wait

end=$(date +%s)
echo "Ended at $(date)"
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_script_for_two_gpu_runs(self, array_size, seed, batch_size,
                                            config_txt, logs_root, cluster,
                                            wandb_project_name, save_to, model_root=None):
        if model_root is None:
            model_root = "results/optimizer=experimental/seed=0"

        train_data = ""

        step_jobs_logs_dir = Path(self.workspace_dir) / "train" / Path(save_to).parent
        step_jobs_logs_dir = str(step_jobs_logs_dir)

        account = self._account()
        python_ws_paths = self._slurm_ws_python_path_setup()

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={self.cpus_per_task * 4}
#SBATCH --job-name={self.job_name}
#SBATCH -p {self.partition}
#SBATCH --time=24:00:00
{account}#SBATCH --array=1-{array_size}

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error  array_job_logs/%x-%A_%a.err

{self.load_env()}

echo "Started at $(date)";
start=`date +%s`

{python_ws_paths}
export WANDB_DIR=${{TRAIN_DIR}}/{wandb_project_name}

LOG_DIR="{step_jobs_logs_dir}"
mkdir -p "${{LOG_DIR}}"

CONFIG_TXT="{config_txt}"
base_line=$(( (SLURM_ARRAY_TASK_ID - 1) * 2 + 1 ))

echo "Array task ${{SLURM_ARRAY_TASK_ID}}"
echo "  Using config lines: ${{base_line}}..$((base_line+3))"

run_one () {{
  local step_idx="$1"
  local line="$2"

  BASE_PORT=28802
  PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID * 10 + step_idx))

  read -r config_ID TRAIN_NUM_SAMPLES LR WD WARMUP BETA1 BETA2 EPS EPOCHS <<< "${{line}}"
  TRAIN_NUM_SAMPLES=${{TRAIN_NUM_SAMPLES%.*}}
  if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
        echo "Error: EPOCHS is not an integer: $EPOCHS"
        exit 1
  fi

  EPOCHS=$((EPOCHS))

  echo "About to start srun for step ${{step_idx}} (config_ID=${{config_ID}}) at $(date)"

  srun --exclusive --ntasks=1 --cpus-per-task={self.cpus_per_task * 2} --gpus-per-task=2 \\
    --output "${{LOG_DIR}}/array_job_logs/step_${{step_idx}}_%x-%A_%a.out" \\
    --error  "${{LOG_DIR}}/array_job_logs/step_${{step_idx}}_%x-%A_%a.err" \\
    torchrun --nproc_per_node 2 --rdzv_endpoint=localhost:${{PORT}} -m clip_train.main -- \\
        --train-data '{train_data}' \\
        --local-loss \\
        --gather-with-grad \\
        --dataset-type webdataset \\
        --precision amp \\
        --report-to wandb \\
        --wandb-project-name '{wandb_project_name}' \\
        --model "local-dir:{model_root}/model_cfg/config_${{config_ID}}" \\
        --train-num-samples "${{TRAIN_NUM_SAMPLES}}" \\
        --epochs ${{EPOCHS}} \\
        --name "train_config${{config_ID}}" \\
        --logs "{logs_root}" \\
        --seed {seed} \\
        --workers {self.cpus_per_task} \\
        --batch-size {batch_size} \\
        --pipeline-id "${{config_ID}}" \\
        --lr "${{LR}}" \\
        --wd "${{WD}}" \\
        --warmup-fraction "${{WARMUP}}" \\
        --beta1 "${{BETA1}}" \\
        --beta2 "${{BETA2}}" \\
        --eps "${{EPS}}" &
    }}

for step_idx in 0 1; do
  line_no=$((base_line + step_idx))
  line=$(sed -n "${{line_no}}p" "${{CONFIG_TXT}}")

  if [ -z "${{line}}" ]; then
    echo "No line ${{line_no}} for array task ${{SLURM_ARRAY_TASK_ID}}, skipping"
    continue
  fi

  echo "  Line${{step_idx}}: ${{line}}"
  run_one "${{step_idx}}" "${{line}}"
done

wait

end=$(date +%s)
echo "Ended at $(date)"
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def _account(self) -> str:
        if self.cluster.lower() == "juwels":
            return "#SBATCH --account=projectnucleus\n"
        return ""

    def _slurm_ws_python_path_setup(self) -> str:
        return f"""
COMMAND_DIR="{self.command_dir}"
TRAIN_DIR="{self.workspace_dir}/train"

export PYTHONPATH="${{COMMAND_DIR}}:${{PYTHONPATH}}"
cd "${{TRAIN_DIR}}"
"""

    def _nccl_env(self) -> str:

        if self.cluster == "meta":
            return """\
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker,virbr
export NCCL_DEBUG=WARN
"""

        return ""
