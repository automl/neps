"""ScaleFM: A framework for scaling studies of deep learning pipelines.
This module provides a unified interface for defining search spaces, running optimizers
under a compute budget, and fitting and plotting the scaling laws of the results.
"""

import logging

from scalefm.api import (
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
from scalefm.optimizers import algorithms
from scalefm.optimizers.ask_and_tell import AskAndTell
from scalefm.optimizers.optimizer import SampledConfig
from scalefm.plot.tensorboard_eval import tblogger
from scalefm.space import SearchSpace
from scalefm.space.scalefm_spaces.parameters import (
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
from scalefm.state import BudgetInfo, Trial
from scalefm.state.pipeline_eval import UserResultDict
from scalefm.status.status import status
from scalefm.utils import convert_operation_to_callable
from scalefm.utils.files import load_and_merge_yamls

# As a library, ScaleFM does not configure logging: no handlers, no levels, no format.
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
