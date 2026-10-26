"""Samples configs for the LLM pre-training search space but never trains anything.
`evaluate_pipeline` returning `None` tells NePS "this trial is being handled
asynchronously"
"""

import logging
from pathlib import Path

import neps


def evaluate_pipeline(**config):
    return None


class HPOSpace(neps.PipelineSpace):
    lr = neps.Float(lower=1e-4, upper=1e-2, log=True, prior=3e-3, prior_confidence="medium")
    weight_decay = neps.Float(lower=1e-3, upper=0.2, log=True, prior=0.1, prior_confidence="medium")
    d_model = neps.Categorical(choices=(128, 256, 384, 512), prior=1, prior_confidence="medium")
    n_heads = neps.Categorical(choices=(2, 4, 8), prior=1, prior_confidence="medium")
    n_layers = neps.Integer(lower=2, upper=8, prior=4, prior_confidence="medium")
    n_tokens = neps.Categorical(
        choices=(5_000_000, 10_000_000, 20_000_000, 50_000_000), prior=2, prior_confidence="medium"
    )


ROOT_DIRECTORY = Path(__file__).resolve().parent / "results" / "hpo_llm_pretraining"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory=ROOT_DIRECTORY,
        optimizer=("random_search", {"use_priors": True}),
        worker_evaluations_to_spend=40,
        ignore_errors=True,
    )
