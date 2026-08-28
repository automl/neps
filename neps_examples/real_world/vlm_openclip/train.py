"""Trains and evaluates one OpenCLIP config sampled by `generate_configs.py`.

Run standalone (async, not through neps), once per config, by a Slurm array
task launched from `array_job.py`. Given either a `configs/config_<id>/`
directory directly, or a group file plus an index into it (how the generated
array jobs call it), it reads that trial's `config.yaml`, trains on the LAION
CC12M image/caption cache prepared by `download_data.py`, and reports the
result directly back to NePS with `save_pipeline_results`.

The objective is the contrastive (CLIP) loss on a held-out slice of the same
pre-training distribution -- the quantity CLIP training actually optimizes.
CIFAR never appears here: it is kept strictly for the *downstream* zero-shot
evaluation in `post_hoc_downstream_eval.py`, which scores the checkpoint this
script saves to `config_dir/checkpoint.pt` on a benchmark that played no part
in the HPO objective.
"""

import argparse
from pathlib import Path

import torch
import yaml
from open_clip import ClipLoss, get_tokenizer
from torch.utils.data import DataLoader

import neps
from common import (
    DEVICE,
    build_model,
    count_params,
    get_gpu_name,
    get_model_flops,
    get_peak_gpu_memory_mb,
    load_data,
)

# #CHANGE_ME: how much of the LAION cache each HPO trial trains on. Must be
# covered by the cache built with `download_data.py --n_samples ...`.
N_TRAIN = 20_000
N_VAL = 2_000

# Validation runs at a fixed batch size regardless of the trial's training
# batch size: the contrastive loss depends on how many negatives are in the
# batch, so a config-dependent val batch size would make trials incomparable.
VAL_BATCH_SIZE = 256

NUM_WORKERS = 4


def contrastive_eval(model, val_loader, loss_fn):
    """Mean contrastive loss and in-batch image->text retrieval accuracy.

    Only full `VAL_BATCH_SIZE` batches are scored, so every batch contributes
    the same number of negatives and the numbers stay comparable across trials.
    """
    model.eval()
    total_loss, correct, total, n_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, texts in val_loader:
            if images.shape[0] < VAL_BATCH_SIZE:
                continue
            images, texts = images.to(DEVICE), texts.to(DEVICE)
            out = model(images, texts)
            total_loss += sum(loss_fn(**out, output_dict=True).values()).item()
            n_batches += 1

            logits = out["logit_scale"] * out["image_features"] @ out["text_features"].t()
            target = torch.arange(images.shape[0], device=DEVICE)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += images.shape[0]

    if n_batches == 0:
        raise RuntimeError(
            f"No full validation batch of {VAL_BATCH_SIZE}; increase N_VAL or lower VAL_BATCH_SIZE."
        )
    return total_loss / n_batches, correct / total


def evaluate(
    lr, wd, vision_width, vision_layers, text_width, text_layers, epoch, batch_size=64, checkpoint_path=None,
):
    tokenizer = get_tokenizer("ViT-B-32")
    train_set, val_set = load_data(tokenizer, n_train=N_TRAIN, n_val=N_VAL)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    model = build_model(vision_width, vision_layers, text_width, text_layers).to(DEVICE)
    loss_fn = ClipLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, **loader_kwargs)

    n_train_samples = 0
    model.train()
    for _ in range(epoch):
        for images, texts in train_loader:
            images, texts = images.to(DEVICE), texts.to(DEVICE)
            optimizer.zero_grad()
            out = model(images, texts)
            loss = sum(loss_fn(**out, output_dict=True).values())
            loss.backward()
            optimizer.step()
            n_train_samples += images.shape[0]

    flops = get_model_flops(model) * n_train_samples
    n_params = count_params(model)
    gpu_name = get_gpu_name()

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    val_loss, val_retrieval_acc = contrastive_eval(model, val_loader, loss_fn)
    peak_gpu_memory_mb = get_peak_gpu_memory_mb()

    return {
        "objective_to_minimize": val_loss,
        "cost": epoch,
        "info_dict": {
            "val_retrieval_acc": val_retrieval_acc,
            "n_train_samples": n_train_samples,
            "flops": flops,
            "n_params": n_params,
            "gpu_name": gpu_name,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--config_dir", type=Path)
    selector.add_argument("--group_file", type=Path)
    parser.add_argument("--task_id", type=int, help="Index into --group_file")
    args = parser.parse_args()

    if args.group_file is not None:
        config_id = yaml.safe_load(args.group_file.read_text())[args.task_id]
        config_dir = args.root_dir / "configs" / f"config_{config_id}"
    else:
        config_dir = args.config_dir

    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    pipeline_id = config_dir.name.removeprefix("config_")

    try:
        result = evaluate(**config, checkpoint_path=config_dir / "checkpoint.pt")
    except Exception as e:  # noqa: BLE001
        result = {"exception": e}

    neps.save_pipeline_results(
        user_result=result,
        pipeline_id=pipeline_id,
        root_directory=args.root_dir,
    )


if __name__ == "__main__":
    main()
