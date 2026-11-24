# Real World Examples

- [TabPFN Scaling Study on ScAn-Bench](#tabpfn-scaling-study-on-scan-bench)
- [OpenCLIP Scaling Study](#openclip-scaling-study)

## **TabPFN Scaling Study on ScAn-Bench**

A scaling study of TabPFN on the ScAn-Bench surrogate benchmark, which needs the `scan_bench` package. Random search samples model and data sizes under a FLOP budget, with a `cost_estimator` that gives the FLOPs of a configuration before it runs. Each evaluation reports its validation loss, its FLOPs, its number of parameters and data, and the NLL on real data as a downstream performance. `scalefm.analyze()` and `scalefm.predict()` then fit the scaling law and predict the loss at larger budgets.

```bash
python tabpfn_scan_bench.py
```

## **OpenCLIP Scaling Study**

A real-world scaling study: how the contrastive loss of an [OpenCLIP](https://github.com/mlfoundations/open_clip) model scales with its tower width and depth and the number of LAION image/caption pairs it is trained on, within a FLOP budget. The runs are laid out by Chinchilla approach 1 or 2, or by ScAn, and extrapolated with `scalefm.predict`. Sampling and training are decoupled so each sampled config -- which can need a very different amount of GPU memory -- is routed to a right-sized Slurm job, with DDP across several GPUs for the largest models.

See [`vlm_openclip/README.md`](https://github.com/automl/scalefm/tree/master/scalefm_examples/real_world/vlm_openclip/README.md) for the full workflow and setup.
