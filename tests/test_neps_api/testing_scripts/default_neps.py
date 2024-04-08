import logging

from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)

import neps
from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping

pipeline_space_fidelity_priors = dict(
    val1=neps.FloatParameter(lower=-10, upper=10, default=1),
    val2=neps.IntegerParameter(lower=1, upper=5, is_fidelity=True),
)

pipeline_space_not_fidelity_priors = dict(
    val1=neps.FloatParameter(lower=-10, upper=10, default=1),
    val2=neps.IntegerParameter(lower=1, upper=5, default=1),
)

pipeline_space_fidelity = dict(
    val1=neps.FloatParameter(lower=-10, upper=10),
    val2=neps.IntegerParameter(lower=1, upper=5, is_fidelity=True),
)

pipeline_space_not_fidelity = dict(
    val1=neps.FloatParameter(lower=-10, upper=10),
    val2=neps.IntegerParameter(lower=1, upper=5),
)


def run_pipeline(val1, val2):
    loss = val1 * val2
    return loss


logging.basicConfig(level=logging.INFO)

# Testing user input "priorband_bo" with argument changes that should be
# accepted in the run.

# Case 1: Choosing priorband

neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_fidelity_priors,
    root_directory="priorband_bo_user_decided",
    max_evaluations_total=1,
    searcher="priorband_bo",
    initial_design_size=5,
    eta=3,
)

# Case 2: Choosing Bayesian optimization

early_hierarchies_considered = "0_1_2_3"
hierarchy_considered = [int(hl) for hl in early_hierarchies_considered.split("_")]
graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
wl_h = [2, 1] + [2] * (len(hierarchy_considered) - 1)
graph_kernels = [
    GraphKernelMapping[kernel](
        h=wl_h[j],
        oa=False,
        se_kernel=None,
    )
    for j, kernel in enumerate(graph_kernels)
]
surrogate_model = ComprehensiveGPHierarchy
surrogate_model_args = {
    "graph_kernels": graph_kernels,
    "hp_kernels": [],
    "verbose": False,
    "hierarchy_consider": hierarchy_considered,
    "d_graph_features": 0,
    "vectorial_features": None,
}
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_not_fidelity,
    root_directory="bo_user_decided",
    max_evaluations_total=1,
    searcher="bayesian_optimization",
    surrogate_model=surrogate_model,
    surrogate_model_args=surrogate_model_args,
)

# Testing neps decision tree on deciding the searcher and rejecting the
# additional arguments.

# Case 1: Choosing priorband
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_fidelity_priors,
    root_directory="priorband_neps_decided",
    max_evaluations_total=1,
    initial_design_size=5,
    eta=3,
)

# Case 2: Choosing bayesian_optimization
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_not_fidelity,
    root_directory="bo_neps_decided",
    max_evaluations_total=1,
)

# Case 3: Choosing pibo
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_not_fidelity_priors,
    root_directory="pibo_neps_decided",
    max_evaluations_total=1,
    initial_design_size=5,
)

# Case 4: Choosing hyperband
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_fidelity,
    root_directory="hyperband_neps_decided",
    max_evaluations_total=1,
    eta=2,
)
