# Interpretable Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel (NAS-BOWL)

This is the code Repository for our proposed method NAS-BOWL.

##  Prerequisites and Dependencies
This code repository contains the submission of NAS-BOWL. To run the codes, please see the prerequisites below:
1. Download the NAS-Bench-101 and NAS-Bench-201 datasets and place the data
files under ```data/``` path. We expect these files:
    
    NAS-Bench-101: ```nasbench_only102.tfrecord```
    
    NAS-Bench-201: ```NAS-Bench-201-v1_1-096897.pth```
    
    and install the relevant NAS-Bench-101 and NAS-Bench-201 APIs.
    
    (N.B. Small RAM machine sometimes has memory issue with this version of NAS-Bench-201. If this is a problem,
    either switch to a large RAM machine or use an earlier version of NAS-Bench-201 (v1.0). If you opt to use the earlier version,
    you have to go to ```./benchmarks/nas201.py``` to change the file name expected, and install a matching version of the NAS-Bench-201
    API, if necessary.)

2. Install the prerequisite packages via ```pip``` or ```conda```. We used Anaconda Python 3.7 for our experiments.
```bash
ConfigSpace==0.4.11
Cython==0.29.16
future==0.18.2
gensim==3.8.0
grakel==0.1b7
graphviz>=0.5.1
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.2
scipy==1.3.1
seaborn==0.9.0
six==1.12.0
statsmodels==0.10.1
tabulate==0.8.3
tensorboard==1.14.0
tensorflow==1.14.0
tensorflow-estimator==1.14.0
torch==1.3.1
tqdm==4.32.1
networkx
```

## Running Experiments
To reproduce the experiments in the paper, see below

1. Search on NAS-Bench-101
    ```bash
    python3 -u nasbench_optimization.py --dataset nasbench101 --pool_size 200 --batch_size 5 --max_iters 30 --n_repeat 20 --n_init 10
    ```

    This by default runs the NAS-Bench-101 on the stochastic validation error (i.e. randomness in the objective function). For the 
    deterministic version, append ```--fixed_query_seed 3``` to the command above.

2. Search on NAS-Bench-201 (by default on the CIFAR-10 valid dataset.)
    ```bash
    python3 -u nasbench_optimization.py  --dataset nasbench201 --pool_size 200 --mutate_size 200 --batch_size 5 --n_init 10 --max_iters 30
    ```
    Again, append ```--fixed_query_seed 3``` for deterministic objective function. Append ```--task cifar100```
    for CIFAR-100 dataset, and similarly ```--task ImageNet16-120``` for ImageNet16 dataset.
    
    For transfer learning results on NAS-Bench-201, run
    ```
    python3 -u transfer_nasbench201.py 
    ```
      
3. To reproduce regression examples on NAS-Bench, use
    ```bash
   # For NAS-Bench-101
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 --dataset nasbench101
   
   # For NAS-Bench-201
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 --dataset nasbench201
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 --dataset nasbench201 --task cifar100
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 --dataset nasbench201 --task ImageNet16-120
    ```
   
4. To run open-domain search on DARTS search space (currently only single-GPU is supported):
    ```bash
    python3 -u run_darts.py --cutout --auxiliary --search_space darts
    ```
   Note that due to the high stochasticity on the CIFAR-10 task, you might not get the exactly same cell from the paper. The code
   above will take approximately 3 days to finish on a single modern GPU. 
   
   This code does not include the evaluation part of the final architecture. However, given the Genotypes returned, 
   you may easily run evaluation using the codes in DARTS repository. See [https://github.com/quark0/darts] for scripts and instructions.
   
5. To reproduce interpretability results (e.g. motifs), we have attached a sample code snippet
in ```./bayesopt/interpreter.py``` (running this file directly will run the code in ```__main__```).

## References
We used materials from the following public code repositories. We would like to express our gratitude towards
these authors/codebase maintainers
    
   1. Ying, Chris, et al. "Nas-bench-101: Towards reproducible neural architecture search." 
   International Conference on Machine Learning (2019). [https://github.com/google-research/nasbench]
   2. White, Colin, Willie Neiswanger, and Yash Savani. "Bananas: Bayesian optimization with neural architectures for neural architecture search." 
   arXiv preprint arXiv:1910.11858 (2019). [https://github.com/naszilla/bananas]
   3. Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." 
   arXiv preprint arXiv:1806.09055 (2018). [https://github.com/quark0/darts]
   4. Dong, Xuanyi, and Yi Yang. "Nas-bench-201: Extending the scope of reproducible neural architecture search." 
   International Conference on Learning Representations (2020). [https://github.com/D-X-Y/NAS-Bench-201]
   5. Siglidis, Giannis, et al. "GraKeL: A Graph Kernel Library in Python." Journal of Machine Learning Research 21.54 (2020): 1-5.
    [https://github.com/ysig/GraKeL]

