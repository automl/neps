#!/bin/bash

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Job to perform
echo "Starting BO-NAS-HPO";
python nasbench_optimization.py --seed 56 --max_iters 100 --n_repeat 3 --optimize_hps --dataset nasbench201 --pool_size 40 --n_init 10 --strategy random
echo "DONE BO-NAS-HPO";
#echo "Starting RND-HPO";
#python nasbench_optimization.py --seed 56 --max_iters 100 --n_repeat 3 --seed 56 --optimize_hps --pool_size 40 --strategy random
#echo "DONE RND-HPO";
#echo "Starting RND-NAS";
#python nasbench_optimization.py --seed 56 --max_iters 100 --n_repeat 3 --seed 56 --optimize_arch --pool_size 40 --strategy random
#echo "DONE RND-NAS";
#echo "Starting RND-NAS-HPO";
#python nasbench_optimization.py --seed 56 --max_iters 100 --n_repeat 3 --seed 56 --optimize_hps --optimize_arch --pool_size 40 --strategy random
#echo "DONE RND-NAS-HPO";

exit $?

echo "DONE";
echo "Finished at $(date)";