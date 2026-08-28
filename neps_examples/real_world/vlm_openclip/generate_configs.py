"""Samples configs for the OpenCLIP search space but never trains anything.

`evaluate_pipeline` returning `None` tells NePS "this trial is being handled
asynchronously" -- it writes the sampled config to
`root_directory/configs/config_<id>/config.yaml` and moves on to sampling the
next one, without blocking on training. The actual training happens
out-of-band in `train.py`, dispatched by `array_job.py` across Slurm job
arrays sized to each config's `batch_size`, and reports results back via
`neps.save_pipeline_results`.

`evaluations_to_spend` bounds how many pending configs this process samples
ahead of the eval workers -- see NePS's "Async mode" note on `neps.run`.
"""

import logging

import neps


def evaluate_pipeline(**config):
    return None


class HPOSpace(neps.PipelineSpace):
    lr = neps.Float(lower=1e-4, upper=1e-2, log=True, prior=1e-3, prior_confidence="medium")
    wd = neps.Float(lower=1e-6, upper=1e-2, log=True, prior=1e-4, prior_confidence="medium")
    vision_width = neps.Categorical(choices=(64, 128, 192, 256), prior=1, prior_confidence="medium")
    vision_layers = neps.Integer(lower=2, upper=6, prior=4, prior_confidence="medium")
    text_width = neps.Categorical(choices=(64, 128, 192, 256), prior=1, prior_confidence="medium")
    text_layers = neps.Integer(lower=2, upper=6, prior=4, prior_confidence="medium")
    batch_size = neps.Categorical(choices=(32, 128, 512), prior=1, prior_confidence="medium")
    epoch = neps.IntegerFidelity(lower=1, upper=5)


ROOT_DIRECTORY = "results/hpo_vlm_openclip"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory=ROOT_DIRECTORY,
        optimizer=("random_search", {"ignore_fidelity": "highest_fidelity", "use_priors": True}),
        evaluations_to_spend=40,
    )
