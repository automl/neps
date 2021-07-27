#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH --mem 0 # memory pool for all cores (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 4 # number of cores
#SBATCH -o results/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e results/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J nasbowl-gboca-random # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Job to perform
python nasbench_optimisation.py -bt eval_cost --budget 150 --n_repeat 5 --seed 42 -ps 150 --pool_strategy random --strategy gboca # --cuda
exit $?

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";