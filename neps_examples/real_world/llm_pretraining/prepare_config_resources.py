
"""Probes each sampled config on one GPU and adds `micro_batch_size`,
`grad_accumulation_steps` and `recommended_num_gpus` to its `config.yaml`, as
scan_llm_bench's `scripts/prepare_config_resources.py` does. `array_job.py` then
groups the configs by `recommended_num_gpus`.

Run it after `generate_configs.py`, on one GPU of the type the training jobs will use:

    python prepare_config_resources.py                    # at most 4 GPUs per config
    python prepare_config_resources.py --max-gpus-cap 2
"""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from generate_configs import ROOT_DIRECTORY
from pipeline.common import DEVICE, SEQ_LEN, VOCAB_SIZE, autocast
from pipeline.model import Transformer


def get_recommended_num_gpus(target_global_batch, max_micro_batch_size):
    """Minimum number of GPUs needed to reach target_global_batch without grad accumulation."""
    if max_micro_batch_size <= 0:
        return 1
    return max(1, (target_global_batch + max_micro_batch_size - 1) // max_micro_batch_size)


def get_target_global_batch_for_tokens(n_tokens):
    """Target global batch size, in sequences, from the number of training tokens."""
    if n_tokens < 100_000_000:
        return 16
    if n_tokens < 1_000_000_000:
        return 32
    if n_tokens < 10_000_000_000:
        return 64
    return 128


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def train_one_step(model, inputs, targets):
    """Run one forward + backward + optimizer step."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    loss.backward()
    optimizer.step()


def find_max_batch_size_ascending(model, max_bs=2048, max_memory_fraction=0.88):
    """Probe micro-batch sizes 1, 2, 4, ... up to max_bs until OOM.

    Returns the largest batch size that fits within max_memory_fraction of GPU memory
    (to leave headroom for DDP and reduce fragmentation OOMs).
    """
    total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
    working_bs = None
    bs = 1
    while bs <= max_bs:
        try:
            print(f"  Probing micro_batch_size={bs}...", flush=True)
            _cleanup()
            tokens = torch.randint(0, VOCAB_SIZE, (bs, SEQ_LEN + 1), device=DEVICE)
            train_one_step(model, tokens[:, :-1], tokens[:, 1:])
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            if peak_gb > total_gb * max_memory_fraction:
                print(
                    f"    batch {bs} | peak VRAM: {peak_gb:.2f} GB > "
                    f"{max_memory_fraction * 100:.0f}% of {total_gb:.2f} GB; using previous batch size.",
                    flush=True,
                )
                break
            working_bs = bs
            print(f"    batch {bs} | peak VRAM: {peak_gb:.2f} GB", flush=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"    batch {bs} OOM. Stopping.", flush=True)
            break
        finally:
            _cleanup()
        bs *= 2

    if working_bs is None:
        raise RuntimeError(
            "Even a micro-batch of one sequence does not fit on this GPU."
        )
    return working_bs


def prepare_single_config(config_path, max_gpus_cap):
    """Probe the micro-batch size and write the resource fields into one config.yaml."""
    config = yaml.safe_load(config_path.read_text())
    model = Transformer(
        VOCAB_SIZE, SEQ_LEN, config["d_model"], config["n_layers"], config["n_heads"]
    ).to(DEVICE)

    target_global = get_target_global_batch_for_tokens(config["n_tokens"])

    # Probe on a single GPU: this finds the largest micro_batch_size that fits, up to
    # the target batch.
    micro_bs = find_max_batch_size_ascending(model, max_bs=target_global)

    recommended_gpus = max(1, min(get_recommended_num_gpus(target_global, micro_bs), max_gpus_cap))
    grad_accum = max(
        1,
        (target_global + micro_bs * recommended_gpus - 1) // (micro_bs * recommended_gpus),
    )

    config["micro_batch_size"] = int(micro_bs)
    config["grad_accumulation_steps"] = int(grad_accum)
    config["recommended_num_gpus"] = int(recommended_gpus)
    config_path.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))

    print(
        f"[{config_path}] micro_batch_size={micro_bs}, "
        f"grad_accumulation_steps={grad_accum}, "
        f"recommended_num_gpus={recommended_gpus}, "
        f"target_global_batch={target_global}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-gpus-cap",
        type=int,
        default=4,
        choices=(1, 2, 4),
        help="Upper cap for recommended_num_gpus (default: 4); one of the num_gpus "
        "entries in resource_map.json.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; prepare_config_resources.py must run on a GPU node "
            "so it can probe micro_batch_size."
        )

    config_paths = [
        path
        for path in sorted(Path(ROOT_DIRECTORY).glob("configs/config_*/config.yaml"))
        if "recommended_num_gpus" not in yaml.safe_load(path.read_text())
    ]
    print(f"Found {len(config_paths)} configs to prepare under {ROOT_DIRECTORY}")
    for config_path in config_paths:
        try:
            prepare_single_config(config_path, args.max_gpus_cap)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {config_path} due to error: {e}")


if __name__ == "__main__":
    main()
