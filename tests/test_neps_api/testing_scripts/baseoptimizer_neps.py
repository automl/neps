import logging
from warnings import warn

import neps
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.optimizers.multi_fidelity.hyperband import Hyperband
from neps.search_spaces.search_space import SearchSpace

pipeline_space_fidelity = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.Integer(lower=1, upper=5, is_fidelity=True),
)

pipeline_space = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.Integer(lower=1, upper=5),
)


def run_pipeline(val1, val2):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(val1, val2)

def evaluate_pipeline(val1, val2):
    loss = val1 * val2
    return loss


def run_pipeline_fidelity(val1, val2):
    warn("run_pipeline_fidelity is deprecated, use evaluate_pipeline_fidelity instead", DeprecationWarning)
    return evaluate_pipeline_fidelity(val1, val2)

def evaluate_pipeline_fidelity(val1, val2):
    loss = val1 * val2
    return {"loss": loss, "cost": 1}


logging.basicConfig(level=logging.INFO)

# Case 1: Testing BaseOptimizer as searcher with Bayesian Optimization
search_space = SearchSpace(**pipeline_space)
my_custom_searcher_1 = BayesianOptimization(
    pipeline_space=search_space, initial_design_size=5
)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="bo_custom_created",
    max_evaluations_total=1,
    searcher=my_custom_searcher_1,
)

# Case 2: Testing BaseOptimizer as searcher with Hyperband
search_space_fidelity = SearchSpace(**pipeline_space_fidelity)
my_custom_searcher_2 = Hyperband(pipeline_space=search_space_fidelity, budget=1)
neps.run(
    evaluate_pipeline=evaluate_pipeline_fidelity,
    root_directory="hyperband_custom_created",
    max_cost_total=1,
    searcher=my_custom_searcher_2,
)
