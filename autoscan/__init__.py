"""AutoScAn: A framework for Neural Architecture Search and Hyperparameter Optimization.
This module provides a unified interface for defining search spaces, running optimizers,
and visualizing results. It includes various optimizers, search space definitions,
and plotting utilities, making it easy to experiment with different configurations
and algorithms.
"""

import logging

from autoscan.api import (
    analyze,
    create_config,
    import_trials,
    load_config,
    load_optimizer_info,
    load_pipeline_space,
    run,
    save_pipeline_results,
)
from autoscan.optimizers import algorithms
from autoscan.optimizers.ask_and_tell import AskAndTell
from autoscan.optimizers.optimizer import SampledConfig
from autoscan.plot.plot import plot
from autoscan.plot.tensorboard_eval import tblogger
from autoscan.space import HPOCategorical, HPOConstant, HPOFloat, HPOInteger, SearchSpace
from autoscan.space.autoscan_spaces.parameters import (
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
from autoscan.state import BudgetInfo, Trial
from autoscan.state.pipeline_eval import UserResultDict
from autoscan.status.status import status
from autoscan.utils import convert_operation_to_callable
from autoscan.utils.files import load_and_merge_yamls

# As a library, AutoScAn does not configure logging: no handlers, no levels, no format.
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
    "analyze",
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
