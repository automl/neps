"""Lays out the experiments of an OpenCLIP scaling study, without training anything.

The model size (`width`, `layers`) and the amount of data (`n_samples`) are the
compute-scaling parameters. A scaling optimizer decides which (model, data) pairs to
train within a FLOP budget. With `is_async=True` and no `evaluate_pipeline`,
`scalefm.acquire` only samples them; `array_job.py` hands each one off to a Slurm job
instead of training it here.

    python generate_configs.py                              # Chinchilla ISO-FLOP
    python generate_configs.py --optimizer chinchilla1      # Chinchilla approach 1
    python generate_configs.py --optimizer scan --target_flops 1e19

The Chinchilla approaches lay out their whole grid up front. ScAn picks each config
from the results so far, so it samples in rounds: the first round is its initial
design (one config per searchable parameter), and every later round needs at least
one result back. Train a round, then run this again for the next one.

Running it again against the same root keeps sampling within the same global
`total_cost_to_spend`, so several generators (or a generator that is restarted once
results come back) share one budget.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import scalefm

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

# Tokens a sample shows the model: 14 x 14 image patches plus a 77 token caption
TOKENS_PER_SAMPLE = (112 // 8) ** 2 + 77


class ScalingSpace(scalefm.PipelineSpace):
    """Model size and data are scaled, the optimization hyperparameters are tuned
    jointly with them (for ScAn)."""

    # Both towers share a width and a depth, which keeps the scaling grid small
    width = scalefm.Integer(64, 512, log=True, log_base=2, is_compute_scaling=True)
    layers = scalefm.Integer(2, 12, is_compute_scaling=True)
    # Samples seen during training, passes over the cache repeat when it is smaller
    n_samples = scalefm.Integer(
        2**13, 2**19, log=True, log_base=2, is_compute_scaling=True
    )
    lr = scalefm.Float(1e-4, 1e-2, log=True)
    wd = scalefm.Float(1e-6, 1e-2, log=True)
    batch_size = scalefm.Integer(32, 512, log=True, log_base=2)


class RecipeSpace(scalefm.PipelineSpace):
    """Model size and data are scaled with a fixed training recipe, as in the
    Chinchilla paper (for the Chinchilla approaches, which do not tune the rest)."""

    width = scalefm.Integer(64, 512, log=True, log_base=2, is_compute_scaling=True)
    layers = scalefm.Integer(2, 12, is_compute_scaling=True)
    n_samples = scalefm.Integer(
        2**13, 2**19, log=True, log_base=2, is_compute_scaling=True
    )
    lr = 1e-3
    wd = 1e-4
    batch_size = 256


def params_estimator(width: int, layers: int, **_) -> float:
    """The parameter count of a config, before training it."""
    from common import model_stats

    return model_stats(int(width), int(layers))[0]


def cost_estimator(width: int, layers: int, n_samples: int, **_) -> float:
    """The training FLOPs of a config, before training it."""
    from common import model_stats

    return model_stats(int(width), int(layers))[1] * n_samples


OPTIMIZERS = {
    # Chinchilla approach 2: configs on ISO-FLOP levels
    "chinchilla2": (
        RecipeSpace,
        (
            "chinchilla",
            {
                "strategy": "approach2",
                "data_parameter_name": "n_samples",
                "params_estimator": params_estimator,
            },
        ),
    ),
    # Chinchilla approach 1: each model on a few data-to-parameter ratios. The ratios
    # are in samples per parameter, a sample holds TOKENS_PER_SAMPLE tokens
    "chinchilla1": (
        RecipeSpace,
        (
            "chinchilla",
            {
                "strategy": "approach1",
                "data_parameter_name": "n_samples",
                "params_estimator": params_estimator,
                "data_multipliers": [0.0025, 0.005, 0.01, 0.02],
            },
        ),
    ),
}


def scan_optimizer(target_flops: float) -> tuple[str, dict]:
    """ScAn: Bayesian optimization that spends the budget on what predicts the loss at
    `target_flops` best."""
    return (
        "bayesian_optimization",
        {
            "acquisition_func_type": "IT-SCAN-JES",
            "target_cost": target_flops,
            "prior_mean": "power_law",
            "budget_schedule": "space_expansion",
        },
    )


ROOT_DIRECTORY = "results/scaling_vlm_openclip"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", choices=[*OPTIMIZERS, "scan"], default="chinchilla2"
    )
    parser.add_argument(
        "--budget", type=float, default=2e17, help="Total training FLOPs to spend."
    )
    parser.add_argument(
        "--max_cost_per_config",
        type=float,
        default=None,
        help="FLOPs of the largest single run, a tenth of the budget by default.",
    )
    parser.add_argument(
        "--target_flops",
        type=float,
        default=1e19,
        help="The compute to extrapolate to, used by ScAn.",
    )
    parser.add_argument(
        "--n_configs",
        type=int,
        default=None,
        help="Configs to sample in this round. By default the initial design for ScAn"
        " (6 configs) and the whole grid for the Chinchilla approaches.",
    )
    parser.add_argument("--root_directory", default=ROOT_DIRECTORY)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    n_configs = args.n_configs
    if args.optimizer == "scan":
        space, optimizer = ScalingSpace(), scan_optimizer(args.target_flops)
        n_configs = n_configs or len(space.get_attrs())
    else:
        space_cls, optimizer = OPTIMIZERS[args.optimizer]
        space = space_cls()

    scalefm.acquire(
        pipeline_space=space,
        root_directory=args.root_directory,
        is_async=True,
        optimizer=optimizer,
        cost_estimator=cost_estimator,
        max_cost_per_config=args.max_cost_per_config or args.budget / 10,
        total_cost_to_spend=args.budget,
        worker_evaluations_to_spend=n_configs,
    )
