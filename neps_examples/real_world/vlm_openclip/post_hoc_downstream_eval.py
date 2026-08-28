"""Post-hoc downstream evaluation for finished `train.py` configs.

The HPO objective in `train.py` only ever sees the contrastive loss on held-out
LAION pre-training data. Here we additionally score each checkpoint zero-shot
on CIFAR-100 -- a benchmark it never trained on and was never tuned against.
The result is written only to `report_down.yaml`, leaving NePS's own
`report.yaml` untouched.

Run once training has produced some finished configs:

    python post_hoc_downstream_eval.py --root_dir results/hpo_vlm_openclip
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from open_clip import get_tokenizer

from torch.utils.data import DataLoader

from common import (
    CIFAR100_CLASSES,
    DEVICE,
    CifarClipDataset,
    build_model,
    image_transform,
    load_cifar100_test,
)


def zero_shot_downstream_eval(model, tokenizer, transform, n_test=1000, batch_size=64):
    test_set = CifarClipDataset(load_cifar100_test(n_test), transform)
    class_tokens = tokenizer([f"a photo of a {name}" for name in CIFAR100_CLASSES]).to(DEVICE)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        text_features = F.normalize(model.encode_text(class_tokens), dim=-1)
        for images, labels in DataLoader(test_set, batch_size=batch_size):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            image_features = F.normalize(model.encode_image(images), dim=-1)
            logits = model.logit_scale.exp() * image_features @ text_features.t()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.numel()

    return {"dataset": "cifar100_zero_shot", "acc1": correct / total, "n_samples": total}


def pending_configs(root_dir):
    for config_dir in sorted(root_dir.glob("configs/config_*")):
        if (
            (config_dir / "report.yaml").exists()
            and (config_dir / "checkpoint.pt").exists()
            and not (config_dir / "report_down.yaml").exists()
        ):
            yield config_dir


def enrich_report(config_dir, downstream_results):
    report_path = config_dir / "report.yaml"
    down_report_path = config_dir / "report_down.yaml"

    state = yaml.safe_load(report_path.read_text()) or {}
    extra = state.get("extra") or {}
    extra["downstream_results"] = downstream_results
    state["extra"] = extra

    down_report_path.write_text(yaml.safe_dump(state, sort_keys=False))


def main(root_dir: Path, n_test: int = 1000):
    transform = image_transform()
    tokenizer = get_tokenizer("ViT-B-32")

    for config_dir in pending_configs(root_dir):
        config = yaml.safe_load((config_dir / "config.yaml").read_text())

        model = build_model(
            config["vision_width"], config["vision_layers"], config["text_width"], config["text_layers"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(config_dir / "checkpoint.pt", map_location=DEVICE))

        downstream_results = zero_shot_downstream_eval(model, tokenizer, transform, n_test=n_test)
        enrich_report(config_dir, downstream_results)

        print(f"{config_dir.name}: cifar100 zero-shot acc1={downstream_results['acc1']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--n_test", default=1000, type=int)
    args = parser.parse_args()

    main(args.root_dir, n_test=args.n_test)
