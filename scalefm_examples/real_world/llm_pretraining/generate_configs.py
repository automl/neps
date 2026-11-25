"""Samples configs for the LLM pre-training search space but never trains anything.
With `is_async=True` and no `evaluate_pipeline`, `scalefm.acquire()` only samples
them, and the Slurm jobs of `array_job.py` train them.
"""

import logging
from pathlib import Path

from pipeline.common import SEQ_LEN, VOCAB_SIZE
from pipeline.model import training_flops

import scalefm


class ScalingSpace(scalefm.PipelineSpace):
    lr = scalefm.Float(lower=1e-4, upper=1e-2, log=True)
    weight_decay = scalefm.Float(lower=1e-3, upper=0.2, log=True)
    d_model = scalefm.Integer(lower=128, upper=512, log=True, log_base=2)
    n_layers = scalefm.Integer(lower=2, upper=8)
    n_tokens = scalefm.Float(lower=5e6, upper=5e7, log=True)


def estimate_flops(d_model, n_layers, n_tokens):
    return training_flops(d_model, n_layers, n_tokens, VOCAB_SIZE, SEQ_LEN)


ROOT_DIRECTORY = Path(__file__).resolve().parent / "results" / "scaling_llm_pretraining"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scalefm.acquire(
        pipeline_space=ScalingSpace(),
        root_directory=ROOT_DIRECTORY,
        is_async=True,
        optimizer="random_search",
        cost_estimator=estimate_flops,
        worker_evaluations_to_spend=40,
    )