from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np

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
from ..multi_fidelity_prior.utils import compute_config_dist, custom_crossover

TOLERANCE = 1e-2  # 1%
SAMPLE_THRESHOLD = 1000  # num samples to be rejected for increasing hypersphere radius
DELTA_THRESHOLD = 1e-2  # 1%


class SamplingPolicy(ABC):
    """Base class for implementing a sampling strategy for SH and its subclasses"""

    def __init__(self, pipeline_space: SearchSpace, patience: int = 100):
        self.pipeline_space = pipeline_space
        self.patience = patience

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
    ):
        super().__init__(pipeline_space=pipeline_space)

    def sample(self, *args, **kwargs) -> SearchSpace:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=True
        )


class FixedPriorPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH but samples
    a fixed fraction from the prior.
    """

    def __init__(self, pipeline_space: SearchSpace, fraction_from_prior: float = 1):
        super().__init__(pipeline_space=pipeline_space)
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
        inc_type: str = "hypersphere",
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
        """
        super().__init__(pipeline_space=pipeline_space)
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
        prob_weights = [v for _, v in sorted(self.policy_map.items())]
        policy_idx = np.random.choice(range(len(prob_weights)), p=prob_weights)
        policy = sorted(self.policy_map.keys())[policy_idx]

        if policy == "prior":
            print(f"Sampling from prior with weights (i, p, r)={prob_weights}")
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        elif policy == "inc":
            print(f"Sampling from inc with weights (i, p, r)={prob_weights}")

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
                print("No incumbent config found, using default as the incumbent.")

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
                # TODO: could instead crossover inc with a random sample
                # generating a config from the prior
                _config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=user_priors, ignore_fidelity=True
                )
                # injecting hyperparameters from the prior config into the incumbent
                # TODO: ideally lower crossover prob overtime
                config = custom_crossover(inc, _config, crossover_prob=0.5)
            else:
                raise ValueError(
                    f"{self.inc_type} is not in {{'hypersphere', 'gaussian'}}"
                )
        else:
            print(f"Sampling from uniform with weights (i, p, r)={prob_weights}")
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
        use_priors: bool = False,
        surrogate_model: str | Any = "gp",
        domain_se_kernel: str = None,
        hp_kernels: list = None,
        surrogate_model_args: dict = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "random",
        patience: int = 100,
    ):
        super().__init__(pipeline_space=pipeline_space)

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

        if pipeline_space.has_prior:
            self.acquisition = DecayingPriorWeightedAcquisition(
                self.acquisition, log=log_prior_weighted
            )

        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": patience, "pipeline_space": pipeline_space},
        )

        self.use_priors = use_priors
        self.sampling_args: dict = {}

    def update_model(self, train_x, train_y):

        pending_ids = train_y.index[np.where(train_y.isna())[0]].to_numpy()
        observed_ids = np.setdiff1d(np.array(train_y.index), pending_ids)

        if len(pending_ids) > 0:
            # We want to use hallucinated results for the evaluations that have
            # not finished yet. For this we fit a model on the finished
            # evaluations and add these to the other results to fit another model.

            self.surrogate_model.fit(
                train_x[observed_ids].to_list(), train_y[observed_ids].to_list()
            )

            ys, _ = self.surrogate_model.predict(train_x[pending_ids].to_list())

            train_y.loc[pending_ids] = ys.detach().numpy()

        self.surrogate_model.fit(train_x.to_list(), train_y.to_list())
        self.acquisition.set_state(self.surrogate_model, decay_t=len(observed_ids))
        self.acquisition_sampler.set_state(x=train_x, y=train_y)

    def sample(self, *args, **kwargs) -> SearchSpace:
        return self.acquisition_sampler.sample(self.acquisition)
