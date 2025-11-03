"""NePS: A framework for Neural Architecture Search and Hyperparameter Optimization.
This module provides a unified interface for defining search spaces, running optimizers,
and visualizing results. It includes various optimizers, search space definitions,
and plotting utilities, making it easy to experiment with different configurations
and algorithms.
"""

from neps.api import create_config, import_trials, run, save_pipeline_results
from neps.optimizers import algorithms
from neps.optimizers.ask_and_tell import AskAndTell
from neps.optimizers.optimizer import SampledConfig
from neps.plot.plot import plot
from neps.plot.tensorboard_eval import tblogger
from neps.space import HPOCategorical, HPOConstant, HPOFloat, HPOInteger, SearchSpace
from neps.space.neps_spaces.parameters import (
    Categorical,
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    Operation,
    PipelineSpace,
    Resampled,
)
from neps.state import BudgetInfo, Trial
from neps.state.pipeline_eval import UserResultDict
from neps.status.status import status
from neps.utils.files import load_and_merge_yamls

__all__ = [
    "AskAndTell",
    "BudgetInfo",
    "Categorical",
    "ConfidenceLevel",
    "Fidelity",
    "Float",
    "HPOCategorical",
    "HPOConstant",
    "HPOFloat",
    "HPOInteger",
    "Integer",
    "Operation",
    "PipelineSpace",
    "Resampled",
    "SampledConfig",
    "SearchSpace",
    "Trial",
    "UserResultDict",
    "algorithms",
    "create_config",
    "import_trials",
    "load_and_merge_yamls",
    "plot",
    "run",
    "save_pipeline_results",
    "status",
    "tblogger",
]
