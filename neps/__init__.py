"""NePS: A framework for Neural Architecture Search and Hyperparameter Optimization.
This module provides a unified interface for defining search spaces, running optimizers,
and visualizing results. It includes various optimizers, search space definitions,
and plotting utilities, making it easy to experiment with different configurations
and algorithms.
"""

from neps.api import run
from neps.optimizers import algorithms, neps_algorithms
from neps.optimizers.ask_and_tell import AskAndTell
from neps.optimizers.optimizer import SampledConfig
from neps.plot.plot import plot
from neps.plot.tensorboard_eval import tblogger
from neps.space import Categorical, Constant, Float, Integer, SearchSpace
from neps.state import BudgetInfo, Trial
from neps.status.status import status
from neps.utils.files import load_and_merge_yamls as load_yamls

__all__ = [
    "AskAndTell",
    "BudgetInfo",
    "Categorical",
    "Constant",
    "Float",
    "Integer",
    "SampledConfig",
    "SearchSpace",
    "Trial",
    "algorithms",
    "load_yamls",
    "neps_algorithms",
    "plot",
    "run",
    "status",
    "tblogger",
]
