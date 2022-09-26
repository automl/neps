import typing
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
from typing_extensions import Literal

from ...search_spaces.hyperparameters.categorical import (
    CATEGORICAL_CONFIDENCE_SCORES,
    CategoricalParameter,
)
from ...search_spaces.hyperparameters.constant import ConstantParameter
from ...search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES, FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy, RandomUniformPolicy
from ..multi_fidelity_prior.v3 import OurOptimizerV3_2
from ..multi_fidelity_prior.v5 import OurOptimizerV5, OurOptimizerV5_2, OurOptimizerV5_3

CAT_CONF_BOUNDS = [1, 8]  # low-to-high
FLOAT_CONF_BOUNDS = [0.05, 0.95]  # high-to-low


class OurOptimizerV6(OurOptimizerV3_2):
    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # checking if the rung to sample at has `eta` configs
        nevals_rung = (self.observed_configs.rung == rung).sum()
        # TODO: the second AND condition is to ensure enough prior samples at higher
        #  rungs, could undo it if we move around priors
        if (
            len(self.observed_configs.rung.unique()) >= self.max_rung + 1
            and nevals_rung >= self.eta - 1
        ):
            # until `eta` configs seen, sample from peaky prior
            cat_conf = np.random.uniform(*CAT_CONF_BOUNDS)
            num_conf = np.random.uniform(*FLOAT_CONF_BOUNDS)
        else:
            # relaxing peakingess when `eta` configs seen at a rung
            cat_conf = CAT_CONF_BOUNDS[1]  # taking the highest confidence
            num_conf = FLOAT_CONF_BOUNDS[0]  # taking the highest confidence
        confidence_score = {"categorical": cat_conf, "numeric": num_conf}
        self._enhance_priors(confidence_score)
        return super().sample_new_config(rung=rung)


class OurOptimizerV6_V5(OurOptimizerV5):
    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # checking if the rung to sample at has `eta` configs
        nevals_rung = (self.observed_configs.rung == rung).sum()
        # TODO: the second AND condition is to ensure enough prior samples at higher
        #  rungs, could undo it if we move around priors
        if (
            len(self.observed_configs.rung.unique()) >= self.max_rung + 1
            and nevals_rung >= self.eta - 1
        ):
            # until `eta` configs seen, sample from peaky prior
            cat_conf = np.random.uniform(*CAT_CONF_BOUNDS)
            num_conf = np.random.uniform(*FLOAT_CONF_BOUNDS)
        else:
            # relaxing peakingess when `eta` configs seen at a rung
            cat_conf = CAT_CONF_BOUNDS[1]  # taking the highest confidence
            num_conf = FLOAT_CONF_BOUNDS[0]  # taking the highest confidence
        confidence_score = {"categorical": cat_conf, "numeric": num_conf}
        self._enhance_priors(confidence_score)
        return super().sample_new_config(rung=rung)


class OurOptimizerV6_V5_2(OurOptimizerV5_2):
    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # checking if the rung to sample at has `eta` configs
        nevals_rung = (self.observed_configs.rung == rung).sum()
        # TODO: the second AND condition is to ensure enough prior samples at higher
        #  rungs, could undo it if we move around priors
        if (
            len(self.observed_configs.rung.unique()) >= self.max_rung + 1
            and nevals_rung >= self.eta - 1
        ):
            # until `eta` configs seen, sample from peaky prior
            cat_conf = np.random.uniform(*CAT_CONF_BOUNDS)
            num_conf = np.random.uniform(*FLOAT_CONF_BOUNDS)
        else:
            # relaxing peakingess when `eta` configs seen at a rung
            cat_conf = CAT_CONF_BOUNDS[1]  # taking the highest confidence
            num_conf = FLOAT_CONF_BOUNDS[0]  # taking the highest confidence
        confidence_score = {"categorical": cat_conf, "numeric": num_conf}
        self._enhance_priors(confidence_score)
        return super().sample_new_config(rung=rung)


class OurOptimizerV6_V5_3(OurOptimizerV5_3):
    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # checking if the rung to sample at has `eta` configs
        nevals_rung = (self.observed_configs.rung == rung).sum()
        # TODO: the second AND condition is to ensure enough prior samples at higher
        #  rungs, could undo it if we move around priors
        if (
            len(self.observed_configs.rung.unique()) >= self.max_rung + 1
            and nevals_rung >= self.eta - 1
        ):
            # until `eta` configs seen, sample from peaky prior
            cat_conf = np.random.uniform(*CAT_CONF_BOUNDS)
            num_conf = np.random.uniform(*FLOAT_CONF_BOUNDS)
        else:
            # relaxing peakingess when `eta` configs seen at a rung
            cat_conf = CAT_CONF_BOUNDS[1]  # taking the highest confidence
            num_conf = FLOAT_CONF_BOUNDS[0]  # taking the highest confidence
        confidence_score = {"categorical": cat_conf, "numeric": num_conf}
        self._enhance_priors(confidence_score)
        return super().sample_new_config(rung=rung)
