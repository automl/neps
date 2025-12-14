"""NePS: A framework for Neural Architecture Search and Hyperparameter Optimization.
This module provides a unified interface for defining search spaces, running optimizers,
and visualizing results. It includes various optimizers, search space definitions,
and plotting utilities, making it easy to experiment with different configurations
and algorithms.
"""

from neps.api import (
    create_config,
    import_trials,
    load_config,
    load_optimizer_info,
    load_pipeline_space,
    run,
    save_pipeline_results,
)
from neps.optimizers import algorithms
from neps.optimizers.ask_and_tell import AskAndTell
from neps.optimizers.optimizer import SampledConfig
from neps.plot.plot import plot
from neps.plot.tensorboard_eval import tblogger
from neps.space import HPOCategorical, HPOConstant, HPOFloat, HPOInteger, SearchSpace
from neps.space.neps_spaces.parameters import (
    ByName,
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    FloatFidelity,
    Integer,
    IntegerFidelity,
    Operation,
    PipelineSpace,
    Resample,
)
from neps.state import BudgetInfo, Trial
from neps.state.pipeline_eval import UserResultDict
from neps.status.status import status
from neps.utils import convert_operation_to_callable
from neps.utils.files import load_and_merge_yamls

__all__ = [
    "AskAndTell",
    "BudgetInfo",
    "ByName",
    "Categorical",
    "ConfidenceLevel",
    "Fidelity",
    "Float",
    "FloatFidelity",
    "HPOCategorical",
    "HPOConstant",
    "HPOFloat",
    "HPOInteger",
    "Integer",
    "IntegerFidelity",
    "Operation",
    "PipelineSpace",
    "Resample",
    "SampledConfig",
    "SearchSpace",
    "Trial",
    "UserResultDict",
    "algorithms",
    "convert_operation_to_callable",
    "create_config",
    "import_trials",
    "load_and_merge_yamls",
    "load_config",
    "load_optimizer_info",
    "load_pipeline_space",
    "plot",
    "run",
    "save_pipeline_results",
    "status",
    "tblogger",
]
