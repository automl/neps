# Real World Examples

1. **Image Segmentation Pipeline Hyperparameter Optimization**

This example demonstrates how to perform hyperparameter optimization (HPO) for an image segmentation pipeline using NePS. The pipeline consists of a ResNet-50 model to segment images model trained on PASCAL Visual Object Classes (VOC) Dataset (http://host.robots.ox.ac.uk/pascal/VOC/).

We compare the performance of the optimized hyperparameters with the default hyperparameters. using the validation loss achieved on the dataset after training the model with the respective hyperparameters.

```bash
python image_segmentation_pipeline_hpo.py
```

The search space has been set with the priors set to the hyperparameters found in this base example: https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning

We run the HPO process for 188 trials and obtain new set of hyperpamereters that outperform the default hyperparameters.

| Hyperparameter | Prior | Optimized Value |
|----------------|-------|-----------------|
| learning_rate  | 0.02 | 0.006745150778442621 |
| batch_size     | 4 | 5 |
| momentum       | 0.5 | 0.5844767093658447 |
| weight_decay   | 0.0001 | 0.00012664785026572645 |


![Validation Loss Curves](../../doc_images/examples/val_loss_image_segmentation.jpg)

The validation loss achieved on the dataset after training the model with the newly sampled hyperparameters is shown in the figure above.

We compare the validation loss values when the model is trained with the default hyperparameters and the optimized hyperparameters:

Validation Loss with Default Hyperparameters: 0.114094577729702

Validation Loss with Optimized Hyperparameters: 0.0997161939740181

The optimized hyperparameters outperform the default hyperparameters by 12.61%.

2. **OpenCLIP Hyperparameter and Architecture Search**

This example demonstrates HPO for an [OpenCLIP](https://github.com/mlfoundations/open_clip) model, searching over optimization hyperparameters (learning rate, weight decay) as well as the vision/text tower width and depth -- the same kind of scan-space used for real OpenCLIP scaling-law studies, shrunk down to a size that trains in seconds.

The model is pre-trained contrastively on real LAION image/caption pairs in webdataset-shard format -- the same format a production LAION scaling study trains on. `download_data.py` reads those shards once, resizes and re-encodes the images, and writes a compact local parquet cache (~530 MB for 100k pairs), so the training loop only ever decodes small JPEGs instead of full-size ones. It reuses whatever is already on disk: an existing cache is used as-is, otherwise local `.tar` shards (e.g. a LAION-400M copy staged on the cluster, via `NEPS_LAION_SHARDS`) are read with no network access, and only failing both does it stream from the [Hugging Face Hub](https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset). See `vlm_openclip/README.md` for the details.

Sampling and training are decoupled, mirroring how a real scaling-law study spreads GPU-hungry training across Slurm job arrays sized to each config's resource needs, while a single lightweight process drives the search:

- `generate_configs.py` calls `neps.run()` with an `evaluate_pipeline` that just returns `None`. NePS takes this as "evaluated asynchronously": it writes the sampled config (including `batch_size`, one of the searched hyperparameters) to `configs/config_<id>/config.yaml` and immediately moves on to sampling the next one, instead of blocking on training.
- `array_job.py` groups the pending configs by `batch_size` into the resource tiers defined in `resource_map.json` (small/medium/large batch sizes -> progressively bigger GPUs), and writes one Slurm array job per non-empty tier under `results/hpo_vlm_openclip/array_jobs/`. Each array task's config is fixed at generation time (a plain list per tier), so there's no scanning or claiming at run time -- task `i` just trains the `i`-th config in its tier's list.
- `train.py` does the actual work: given a config (directly, or via a tier's group file + task index), it reads `config.yaml`, trains and evaluates that config, and reports the result straight back to NePS with `neps.save_pipeline_results`. It also saves the trained weights to `config_<id>/checkpoint.pt`.

```bash
pip install -r vlm_openclip/requirements.txt
python vlm_openclip/download_data.py
python vlm_openclip/generate_configs.py
python vlm_openclip/array_job.py
sbatch vlm_openclip/results/hpo_vlm_openclip/array_jobs/array_job_small.sh   # and/or medium, large
```

Once some configs have finished training, `post_hoc_downstream_eval.py` runs a *post-hoc* downstream evaluation: for every finished config it loads the saved checkpoint and scores it zero-shot on CIFAR-100 -- a benchmark that played no part in the HPO objective (which only ever sees the contrastive loss on held-out LAION data). This mirrors how a real scaling-law study defers expensive zero-shot/retrieval evals on held-out benchmarks to a separate stage after training, instead of folding them into the search objective itself. The results are written to `config_<id>/report_down.yaml`, alongside (and without touching) NePS's own `report.yaml`.

```bash
python vlm_openclip/post_hoc_downstream_eval.py --root_dir vlm_openclip/results/hpo_vlm_openclip
```
