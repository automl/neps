from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch

from metahyper import instance_from_map

from ...search_spaces.search_space import SearchSpace
from ..bayesian_optimization.acquisition_functions import AcquisitionMapping
from ..bayesian_optimization.acquisition_functions.base_acquisition import BaseAcquisition
from ..bayesian_optimization.acquisition_functions.prior_weighted import (
    DecayingPriorWeightedAcquisition,
)
from ..bayesian_optimization.acquisition_samplers import AcquisitionSamplerMapping
from ..bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from ..bayesian_optimization.kernels.get_kernels import get_kernels
from ..bayesian_optimization.models import SurrogateModelMapping
from ..multi_fidelity_prior.utils import (
    compute_config_dist,
    custom_crossover,
    local_mutation,
    update_fidelity,
)

TOLERANCE = 1e-2  # 1%
SAMPLE_THRESHOLD = 1000  # num samples to be rejected for increasing hypersphere radius
DELTA_THRESHOLD = 1e-2  # 1%
TOP_EI_SAMPLE_COUNT = 10


class SamplingPolicy(ABC):
    """Base class for implementing a sampling strategy for SH and its subclasses"""

    def __init__(self, pipeline_space: SearchSpace, patience: int = 100, logger=None):
        self.pipeline_space = pipeline_space
        self.patience = patience
        self.logger = logger or logging.getLogger("neps")

    @abstractmethod
    def sample(self, *args, **kwargs) -> SearchSpace:
        pass


class RandomUniformPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH / hyperband

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        logger=None,
    ):
        super().__init__(pipeline_space=pipeline_space, logger=logger)

    def sample(self, *args, **kwargs) -> SearchSpace:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=True
        )


class FixedPriorPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH but samples
    a fixed fraction from the prior.
    """

    def __init__(
        self, pipeline_space: SearchSpace, fraction_from_prior: float = 1, logger=None
    ):
        super().__init__(pipeline_space=pipeline_space, logger=logger)
        assert 0 <= fraction_from_prior <= 1
        self.fraction_from_prior = fraction_from_prior

    def sample(self, *args, **kwargs) -> SearchSpace:
        """Samples from the prior with a certain probabiliyu

        Returns:
            SearchSpace: [description]
        """
        user_priors = False
        if np.random.uniform() < self.fraction_from_prior:
            user_priors = True
        config = self.pipeline_space.sample(
            patience=self.patience, user_priors=user_priors, ignore_fidelity=True
        )
        return config


class EnsemblePolicy(SamplingPolicy):
    """Ensemble of sampling policies including sampling randomly, from prior & incumbent.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        inc_type: str = "mutation",
        logger=None,
    ):
        """Samples a policy as per its weights and performs the selected sampling.

        Args:
            pipeline_space: Space in which to search
            inc_type: str
                if "hypersphere", uniformly samples from around the incumbent within its
                    distance from the nearest neighbour in history
                if "gaussian", samples from a gaussian around the incumbent
                if "crossover", generates a config by crossover between a random sample
                    and the incumbent
                if "mutation", generates a config by perturbing each hyperparameter with
                    50% (mutation_rate=0.5) probability of selecting each hyperparmeter
                    for perturbation, sampling a deviation N(value, mutation_std=0.5))
        """
        super().__init__(pipeline_space=pipeline_space, logger=logger)
        self.inc_type = inc_type
        # setting all probabilities uniformly
        self.policy_map = {"random": 0.33, "prior": 0.34, "inc": 0.33}

    def sample_neighbour(self, incumbent, distance, tolerance=TOLERANCE):
        """Samples a config from around the `incumbent` within radius as `distance`."""
        # TODO: how does tolerance affect optimization on landscapes of different scale
        sample_counter = 0
        while True:
            # sampling a config
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            # computing distance from incumbent
            d = compute_config_dist(config, incumbent)
            # checking if sample is within the hypersphere around the incumbent
            if d < max(distance, tolerance):
                # accept sample
                break
            sample_counter += 1
            if sample_counter > SAMPLE_THRESHOLD:
                # reset counter for next increased radius for hypersphere
                sample_counter = 0
                # if no sample falls within the radius, increase the threshold radius 1%
                distance += distance * DELTA_THRESHOLD
        # end of while
        return config

    def sample(
        self, inc: SearchSpace = None, weights: dict[str, float] = None, *args, **kwargs
    ) -> SearchSpace:
        """Samples from the prior with a certain probability

        Returns:
            SearchSpace: [description]
        """
        if weights is not None:
            for key, value in sorted(weights.items()):
                self.policy_map[key] = value
        else:
            self.logger.info(f"Using default policy weights: {self.policy_map}")
        prob_weights = [v for _, v in sorted(self.policy_map.items())]
        policy_idx = np.random.choice(range(len(prob_weights)), p=prob_weights)
        policy = sorted(self.policy_map.keys())[policy_idx]

        self.logger.info(f"Sampling from {policy} with weights (i, p, r)={prob_weights}")

        if policy == "prior":
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        elif policy == "inc":
            # pylint: disable=simplifiable-if-statement
            if (
                hasattr(self.pipeline_space, "has_prior")
                and self.pipeline_space.has_prior
            ):
                user_priors = True
            else:
                user_priors = False

            if inc is None:
                inc = deepcopy(self.pipeline_space.sample_default_configuration())
                self.logger.warning(
                    "No incumbent config found, using default as the incumbent."
                )

            if self.inc_type == "hypersphere":
                distance = kwargs["distance"]
                config = self.sample_neighbour(inc, distance)
            elif self.inc_type == "gaussian":
                # use inc to set the defaults of the configuration
                _inc = deepcopy(inc)
                _inc.set_defaults_to_current_values()
                # then sample with prior=True from that configuration
                # since the defaults are treated as the prior
                config = _inc.sample(
                    patience=self.patience, user_priors=user_priors, ignore_fidelity=True
                )
            elif self.inc_type == "crossover":
                # choosing the configuration for crossover with incumbent
                # the weight distributed across prior adnd inc
                _w_priors = 1 - self.policy_map["random"]
                # re-calculate normalized score ratio for prior-inc
                w_prior = np.clip(self.policy_map["prior"] / _w_priors, a_min=0, a_max=1)
                w_inc = np.clip(self.policy_map["inc"] / _w_priors, a_min=0, a_max=1)
                # calculating difference of prior and inc score
                score_diff = np.abs(w_prior - w_inc)
                # using the difference in score as the weight of what to sample when
                # if the score difference is small, crossover between incumbent and prior
                # if the score difference is large, crossover between incumbent and random
                probs = [1 - score_diff, score_diff]  # the order is [prior, random]
                user_priors = np.random.choice([True, False], p=probs)
                if (
                    hasattr(self.pipeline_space, "has_prior")
                    and not self.pipeline_space.has_prior
                ):
                    user_priors = False
                self.logger.info(
                    f"Crossing over with user_priors={user_priors} with p={probs}"
                )
                # sampling a configuration either randomly or from a prior
                _config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=user_priors, ignore_fidelity=True
                )
                # injecting hyperparameters from the sampled config into the incumbent
                # TODO: ideally lower crossover prob overtime
                config = custom_crossover(inc, _config, crossover_prob=0.5)
            elif self.inc_type == "mutation":
                if "inc_mutation_rate" in kwargs:
                    config = local_mutation(
                        inc,
                        mutation_rate=kwargs["inc_mutation_rate"],
                        std=kwargs["inc_mutation_std"],
                    )
                else:
                    config = local_mutation(inc)
            else:
                raise ValueError(
                    f"{self.inc_type} is not in "
                    f"{{'mutation', 'crossover', 'hypersphere', 'gaussian'}}"
                )
        else:
            # random
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=True
            )
        return config


class ModelPolicy(SamplingPolicy):
    """A policy for sampling configuration, i.e. the default for SH / hyperband

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: str | Any = "gp",
        domain_se_kernel: str = None,
        hp_kernels: list = None,
        surrogate_model_args: dict = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "random",
        patience: int = 100,
        logger=None,
    ):
        super().__init__(pipeline_space=pipeline_space, logger=logger)

        surrogate_model_args = surrogate_model_args or {}

        _, hp_kernels = get_kernels(
            pipeline_space=pipeline_space,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=None,
            hp_kernels=hp_kernels,
            optimal_assignment=False,
        )
        if "graph_kernels" not in surrogate_model_args:
            surrogate_model_args["graph_kernels"] = None
        if "hp_kernels" not in surrogate_model_args:
            surrogate_model_args["hp_kernels"] = hp_kernels
        if not surrogate_model_args["hp_kernels"]:
            raise ValueError("No kernels are provided!")
        if "vectorial_features" not in surrogate_model_args:
            surrogate_model_args[
                "vectorial_features"
            ] = pipeline_space.get_vectorial_dim()

        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs=surrogate_model_args,
        )

        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )

        # TODO: Enable only when a flag exists to toggle prior-based decaying of AF
        # if pipeline_space.has_prior:
        #     self.acquisition = DecayingPriorWeightedAcquisition(
        #         self.acquisition, log=log_prior_weighted
        #     )

        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": patience, "pipeline_space": pipeline_space},
        )

        self.sampling_args: dict = {}

    def _fantasize_pending(self, train_x, train_y, pending_x):
        if len(pending_x) == 0:
            return train_x, train_y
        # fit model on finished evaluations
        self.surrogate_model.fit(train_x, train_y)
        # hallucinating: predict for the pending evaluations
        _y, _ = self.surrogate_model.predict(pending_x)
        _y = _y.detach().numpy().tolist()
        # appending to training data
        train_x.extend(pending_x)
        train_y.extend(_y)
        return train_x, train_y

    def update_model(self, train_x, train_y, pending_x, decay_t=None):
        if decay_t is None:
            decay_t = len(train_x)
        train_x, train_y = self._fantasize_pending(train_x, train_y, pending_x)
        self.surrogate_model.fit(train_x, train_y)
        self.acquisition.set_state(self.surrogate_model, decay_t=decay_t)
        # TODO: set_state should generalize to all options
        #  no needed to set state of sampler when using `random`
        # self.acquisition_sampler.set_state(x=train_x, y=train_y)

    def sample(
        self, active_max_fidelity: int = None, fidelity: int = None, **kwargs
    ) -> SearchSpace:
        """Performs the equivalent of optimizing the acquisition function.

        Performs 2 strategies as per the arguments passed:
            * If fidelity is not None, triggers the case when the surrogate has been
              trained jointly with the fidelity dimension, i.e., all observations ever
              recorded. In this case, the EI for random samples is evaluated at the
              `fidelity` where the new sample will be evaluated. The top-10 are selected,
              and the EI for them is evaluated at the target/mmax fidelity.
            * If active_max_fidelity is not None, triggers the case when a surrogate is
              trained per fidelity. In this case, all samples have their fidelity
              variable set to the same value. This value is same as that of the fidelity
              value of the configs in the training data.
        """
        self.logger.info("Acquiring...")

        # sampling random configurations
        samples = [
            self.pipeline_space.sample(user_priors=False, ignore_fidelity=True)
            for _ in range(SAMPLE_THRESHOLD)
        ]

        if fidelity is not None:
            # w/o setting this flag, the AF eval will set all fidelities to max
            self.acquisition.optimize_on_max_fidelity = False
            _inc_copy = self.acquisition.incumbent
            # TODO: better design required, for example, not import torch
            #  right now this case handles the 2-step acquisition in `sample`
            if "incumbent" in kwargs:
                # sets the incumbent to the best score at the required fidelity for
                # correct computation of EI scores
                self.acquisition.incumbent = torch.tensor(kwargs["incumbent"])
            # updating the fidelity of the sampled configurations
            samples = list(map(update_fidelity, samples, [fidelity] * len(samples)))
            # computing EI at the given `fidelity`
            eis = self.acquisition.eval(x=samples, asscalar=True)
            # extracting the 10 highest scores
            _ids = np.argsort(eis)[-TOP_EI_SAMPLE_COUNT:]
            samples = pd.Series(samples).iloc[_ids].values.tolist()
            # setting the fidelity to the maximum fidelity
            self.acquisition.optimize_on_max_fidelity = True
            self.acquisition.incumbent = _inc_copy

        if active_max_fidelity is not None:
            # w/o setting this flag, the AF eval will set all fidelities to max
            self.acquisition.optimize_on_max_fidelity = False
            fidelity = active_max_fidelity
            samples = list(map(update_fidelity, samples, [fidelity] * len(samples)))

        # computes the EI for all `samples`
        eis = self.acquisition.eval(x=samples, asscalar=True)
        # extracting the highest scored sample
        config = samples[np.argmax(eis)]
        # TODO: can generalize s.t. sampler works for all types, currently,
        #  random sampler in NePS does not do what is required here
        # return self.acquisition_sampler.sample(self.acquisition)
        return config
