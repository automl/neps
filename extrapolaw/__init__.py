"""Extrapolaw: A framework for scaling studies of deep learning pipelines.
This module provides a unified interface for defining search spaces, running optimizers
under a compute budget, and fitting and plotting the scaling laws of the results.
"""

import logging

from extrapolaw.api import (
    acquire,
    analyze,
    create_config,
    import_trials,
    load_config,
    load_optimizer_info,
    load_pipeline_space,
    predict,
    save_pipeline_results,
)
from extrapolaw.optimizers import algorithms
from extrapolaw.optimizers.ask_and_tell import AskAndTell
from extrapolaw.optimizers.optimizer import SampledConfig
from extrapolaw.plot.tensorboard_eval import tblogger
from extrapolaw.space import SearchSpace
from extrapolaw.space.extrapolaw_spaces.parameters import (
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
from extrapolaw.state import BudgetInfo, Trial
from extrapolaw.state.pipeline_eval import UserResultDict
from extrapolaw.status.status import status
from extrapolaw.utils import convert_operation_to_callable
from extrapolaw.utils.files import load_and_merge_yamls

# As a library, Extrapolaw does not configure logging: no handlers, no levels, no format.
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
    "Integer",
    "IntegerFidelity",
    "Operation",
    "PipelineSpace",
    "Resample",
    "SampledConfig",
    "SearchSpace",
    "Trial",
    "UserResultDict",
    "acquire",
    "algorithms",
    "analyze",
    "convert_operation_to_callable",
    "create_config",
    "import_trials",
    "load_and_merge_yamls",
    "load_config",
    "load_optimizer_info",
    "load_pipeline_space",
    "predict",
    "save_pipeline_results",
    "status",
    "tblogger",
]
