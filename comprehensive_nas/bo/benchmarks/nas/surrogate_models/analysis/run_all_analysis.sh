#!/bin/bash
#SBATCH -p test_cpu-ivy
#SBATCH --array=1-6
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH -o logs/%A-%a.o
#SBATCH -e logs/%A-%a.e
#SBATCH --gres=gpu:0  # reserves no GPUs

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Setup
# Activate conda environment
source ~/.bashrc
conda activate pytorch1.3
# export PYTHONPATH=$PWD

# Activate your conda/venv environment prior to executing this

# Run correlation analysis
if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/correlation_between_fidelities.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3 --configspace_path='configspace.json'
  exit $?
fi

# Run extrapolation
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/extrapolation_to_val_random_configs.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3
  exit $?
fi

# Run interpolation
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/interpolation_analysis.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3
  exit $?
fi

# Increase number of parameter free-operations
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/op_list_parameter_free_op_increase.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3
  exit $?
fi

# Vary fidelities
if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/varying_fidelities.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3
  exit $?
fi

# Vary hyperparameter
if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/varying_hyperparameters.py --model_log_dir=$1 --nasbench_data=/home/anonymous/NasBench301_v0.3
  exit $?
fi
