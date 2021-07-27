#!/bin/bash

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Job to perform
python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset branin2 --n_init 10 --cuda
python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset hartmann3 --n_init 10 --cuda
python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset hartmann6 --n_init 10 --cuda

python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset branin2 --n_init 10 --cuda --strategy random
python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset hartmann3 --n_init 10 --cuda --strategy random
python nasbench_optimization.py --seed 56 --max_iters 200 --n_repeat 5 --optimize_hps --pool_size 40 --dataset hartmann6 --n_init 10 --cuda --strategy random

exit $?

echo "DONE";
echo "Finished at $(date)";