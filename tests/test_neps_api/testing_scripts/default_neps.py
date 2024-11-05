import logging
from warnings import warn

import neps

pipeline_space_fidelity_priors = dict(
    val1=neps.Float(lower=-10, upper=10, default=1),
    val2=neps.Integer(lower=1, upper=5, is_fidelity=True),
)

pipeline_space_not_fidelity_priors = dict(
    val1=neps.Float(lower=-10, upper=10, default=1),
    val2=neps.Integer(lower=1, upper=5, default=1),
)

pipeline_space_fidelity = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.Integer(lower=1, upper=5, is_fidelity=True),
)

pipeline_space_not_fidelity = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.Integer(lower=1, upper=5),
)


def run_pipeline(val1, val2):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(val1, val2)

def evaluate_pipeline(val1, val2):
    loss = val1 * val2
    return loss


logging.basicConfig(level=logging.INFO)

# Testing user input "priorband_bo" with argument changes that should be
# accepted in the run.

# Case 1: Choosing priorband

neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space_fidelity_priors,
    root_directory="priorband_bo_user_decided",
    max_evaluations_total=1,
    searcher="priorband_bo",
    initial_design_size=5,
    eta=3,
)

# Testing neps decision tree on deciding the searcher and rejecting the
# additional arguments.

# Case 1: Choosing priorband
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space_fidelity_priors,
    root_directory="priorband_neps_decided",
    max_evaluations_total=1,
    initial_design_size=5,
    eta=3,
)

# Case 2: Choosing bayesian_optimization
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space_not_fidelity,
    root_directory="bo_neps_decided",
    max_evaluations_total=1,
)

# Case 3: Choosing pibo
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space_not_fidelity_priors,
    root_directory="pibo_neps_decided",
    max_evaluations_total=1,
    initial_design_size=5,
)

# Case 4: Choosing hyperband
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space_fidelity,
    root_directory="hyperband_neps_decided",
    max_evaluations_total=1,
    eta=2,
)
