# Real World Examples

- [TabPFN Scaling Study on ScAn-Bench](#tabpfn-scaling-study-on-scan-bench)
- [OpenCLIP Hyperparameter and Architecture Search](#openclip-hyperparameter-and-architecture-search)

## **TabPFN Scaling Study on ScAn-Bench**

A scaling study of TabPFN on the ScAn-Bench surrogate benchmark, which needs the `scan_bench` package. Random search samples model and data sizes under a FLOP budget, with a `cost_estimator` that gives the FLOPs of a configuration before it runs. Each evaluation reports its validation loss, its FLOPs, its number of parameters and data, and the NLL on real data as a downstream performance. `extrapolaw.analyze()` and `extrapolaw.predict()` then fit the scaling law and predict the loss at larger budgets.

```bash
python tabpfn_scan_bench.py
```

## **OpenCLIP Hyperparameter and Architecture Search**

A real-world HPO example: searching an [OpenCLIP](https://github.com/mlfoundations/open_clip) model's optimization hyperparameters (learning rate, weight decay) together with its vision/text tower width and depth, pre-training contrastively on real LAION image/caption pairs. Sampling and training are decoupled so each sampled config -- which can need a very different amount of GPU memory -- is routed to a right-sized Slurm job instead of blocking one worker on training.

See [`vlm_openclip/README.md`](https://github.com/automl/extrapolaw/tree/master/extrapolaw_examples/real_world/vlm_openclip/README.md) for the full workflow and setup.
