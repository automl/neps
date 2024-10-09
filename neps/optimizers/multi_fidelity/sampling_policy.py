from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    AcquisitionFunction,
    LinearMCObjective,
    qLogNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood

from neps.optimizers.bayesian_optimization.acquisition_functions.pibo import (
    pibo_acquisition,
)
from neps.optimizers.bayesian_optimization.models.gp import make_default_single_obj_gp
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from botorch.acquisition.analytic import SingleTaskGP

    from neps.sampling.priors import Prior
    from neps.search_spaces.search_space import SearchSpace

TOLERANCE = 1e-2  # 1%
SAMPLE_THRESHOLD = 1000  # num samples to be rejected for increasing hypersphere radius
DELTA_THRESHOLD = 1e-2  # 1%
TOP_EI_SAMPLE_COUNT = 10

logger = logging.getLogger(__name__)


def update_fidelity(config: SearchSpace, fidelity: int | float) -> SearchSpace:
    assert config.fidelity is not None
    config.fidelity.set_value(fidelity)
    return config


class SamplingPolicy(ABC):
    """Base class for implementing a sampling strategy for SH and its subclasses."""

    def __init__(self, pipeline_space: SearchSpace, patience: int = 100):
        self.pipeline_space = pipeline_space
        self.patience = patience

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> SearchSpace: ...


class RandomUniformPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH / hyperband.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(self, pipeline_space: SearchSpace):
        super().__init__(pipeline_space=pipeline_space)

    def sample(self, *args: Any, **kwargs: Any) -> SearchSpace:
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

    def sample(self, *args: Any, **kwargs: Any) -> SearchSpace:
        """Samples from the prior with a certain probabiliyu.

        Returns:
            SearchSpace: [description]
        """
        user_priors = False
        if np.random.uniform() < self.fraction_from_prior:
            user_priors = True
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=user_priors, ignore_fidelity=True
        )


class EnsemblePolicy(SamplingPolicy):
    """Ensemble of sampling policies including sampling randomly, from prior & incumbent.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        inc_type: Literal[
            "hypersphere", "gaussian", "crossover", "mutation"
        ] = "mutation",
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
        super().__init__(pipeline_space=pipeline_space)
        self.inc_type = inc_type
        # setting all probabilities uniformly
        self.policy_map = {"random": 0.33, "prior": 0.34, "inc": 0.33}

    def sample_neighbour(
        self,
        incumbent: SearchSpace,
        distance: float,
        tolerance: float = TOLERANCE,
    ) -> SearchSpace:
        """Samples a config from around the `incumbent` within radius as `distance`."""
        # TODO: how does tolerance affect optimization on landscapes of different scale
        sample_counter = 0
        from neps.optimizers.multi_fidelity_prior.utils import (
            compute_config_dist,
        )

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

    def sample(  # noqa: PLR0912, C901
        self,
        inc: SearchSpace | None = None,
        weights: dict[str, float] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SearchSpace:
        """Samples from the prior with a certain probability.

        Returns:
            SearchSpace: [description]
        """
        from neps.optimizers.multi_fidelity_prior.utils import (
            custom_crossover,
            local_mutation,
        )

        if weights is not None:
            for key, value in sorted(weights.items()):
                self.policy_map[key] = value
        else:
            logger.info(f"Using default policy weights: {self.policy_map}")
        prob_weights = [v for _, v in sorted(self.policy_map.items())]
        policy_idx = np.random.choice(range(len(prob_weights)), p=prob_weights)
        policy = sorted(self.policy_map.keys())[policy_idx]

        logger.info(f"Sampling from {policy} with weights (i, p, r)={prob_weights}")

        if policy == "prior":
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        elif policy == "inc":
            if (
                hasattr(self.pipeline_space, "has_prior")
                and self.pipeline_space.has_prior
            ):
                user_priors = True
            else:
                user_priors = False

            if inc is None:
                inc = self.pipeline_space.sample_default_configuration().clone()
                logger.warning(
                    "No incumbent config found, using default as the incumbent."
                )

            if self.inc_type == "hypersphere":
                distance = kwargs["distance"]
                config = self.sample_neighbour(inc, distance)
            elif self.inc_type == "gaussian":
                # use inc to set the defaults of the configuration
                _inc = inc.clone()
                _inc.set_defaults_to_current_values()
                # then sample with prior=True from that configuration
                # since the defaults are treated as the prior
                config = _inc.sample(
                    patience=self.patience,
                    user_priors=user_priors,
                    ignore_fidelity=True,
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
                logger.info(
                    f"Crossing over with user_priors={user_priors} with p={probs}"
                )
                # sampling a configuration either randomly or from a prior
                _config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=user_priors,
                    ignore_fidelity=True,
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
    """A policy for sampling configuration, i.e. the default for SH / hyperband.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        prior: Prior | None = None,
        use_cost: bool = False,
        device: torch.device | None = None,
    ):
        if use_cost:
            raise NotImplementedError("Cost is not implemented yet.")

        super().__init__(pipeline_space=pipeline_space)
        self.device = device
        self.prior = prior
        self._encoder = ConfigEncoder.default(
            {**pipeline_space.numerical, **pipeline_space.categoricals},
            constants=pipeline_space.constants,
        )
        self._model: SingleTaskGP | None = None
        self._acq: AcquisitionFunction | None = None

    def update_model(
        self,
        train_x: list[SearchSpace],
        train_y: list[float],
        pending_x: list[SearchSpace],
        decay_t: float | None = None,
    ) -> None:
        x_train = self._encoder.encode([config.hp_values() for config in train_x])
        x_pending = self._encoder.encode([config.hp_values() for config in pending_x])
        y_train = torch.tensor(train_y, dtype=torch.float64, device=self.device)

        # TODO: Most of this just copies BO and the duplication can be replaced
        # once we don't have the two stage `update_model()` and `sample()`
        y_model = make_default_single_obj_gp(x_train, y_train, encoder=self._encoder)

        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(likelihood=y_model.likelihood, model=y_model),
        )
        acq = qLogNoisyExpectedImprovement(
            y_model,
            X_baseline=x_train,
            X_pending=x_pending,
            # Unfortunatly, there's no option to indicate that we minimize
            # the AcqFunction so we need to do some kind of transformation.
            # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
            objective=LinearMCObjective(weights=torch.tensor([-1.0])),
        )

        # If we have a prior, wrap the above acquisitionm with a prior weighting
        if self.prior is not None:
            assert decay_t is not None
            # TODO: Ideally we have something based on budget and dimensions, not an
            # arbitrary term. This 10 is extracted from the old DecayingWeightedPrior
            pibo_exp_term = 10 / decay_t
            significant_lower_bound = 1e-4  # No significant impact beyond this point
            if pibo_exp_term < significant_lower_bound:
                acq = pibo_acquisition(
                    acq,
                    prior=self.prior,
                    prior_exponent=pibo_exp_term,
                    x_domain=self._encoder.domains,
                )

        self._y_model = y_model
        self._acq = acq

    # TODO: rework with MFBO
    def sample(
        self,
        active_max_fidelity: int | None = None,
        fidelity: int | None = None,
        **kwargs: Any,
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
        return samples[np.argmax(eis)]
