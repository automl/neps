"""Samples configs for the OpenCLIP search space but never trains anything.

`evaluate_pipeline` returning `None` tells AutoScAn "this trial is being handled
asynchronously"
"""

import logging

import autoscan


def evaluate_pipeline(**config):
    return None


class HPOSpace(autoscan.PipelineSpace):
    lr = autoscan.Float(lower=1e-4, upper=1e-2, log=True, prior=1e-3, prior_confidence="medium")
    wd = autoscan.Float(lower=1e-6, upper=1e-2, log=True, prior=1e-4, prior_confidence="medium")
    vision_width = autoscan.Categorical(choices=(64, 128, 192, 256), prior=1, prior_confidence="medium")
    vision_layers = autoscan.Integer(lower=2, upper=6, prior=4, prior_confidence="medium")
    text_width = autoscan.Categorical(choices=(64, 128, 192, 256), prior=1, prior_confidence="medium")
    text_layers = autoscan.Integer(lower=2, upper=6, prior=4, prior_confidence="medium")
    batch_size = autoscan.Categorical(choices=(32, 128, 512), prior=1, prior_confidence="medium")
    epoch = autoscan.IntegerFidelity(lower=1, upper=5)


ROOT_DIRECTORY = "results/hpo_vlm_openclip"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    autoscan.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory=ROOT_DIRECTORY,
        optimizer=("random_search", {"ignore_fidelity": "highest_fidelity", "use_priors": True}),
        worker_evaluations_to_spend=40,
    )
