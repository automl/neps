from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
)
from neps.optimizers.bayesian_optimization.kernels.get_kernels import get_kernels
from neps.optimizers.bayesian_optimization.models import SurrogateModelMapping
from neps.search_spaces.samplers.sampler import Sampler
from neps.search_spaces.samplers.uniform import UniformSampler
from neps.utils.common import instance_from_map

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
        BaseAcquisition,
    )
    from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
        AcquisitionSampler,
    )
    from neps.search_spaces import SearchSpace
    from neps.utils.types import Number


class ModelPolicy(Sampler):
    """A policy for sampling configuration, i.e. the default for SH / hyperband.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        *,
        space: SearchSpace,
        surrogate_model: str | Any = "gp",
        surrogate_model_args: Mapping[str, Any] | None = None,
        domain_se_kernel: str | None = None,
        graph_kernels: list | None = None,
        hp_kernels: list | None = None,
        acquisition: str | BaseAcquisition | type[BaseAcquisition] = "EI",
        acquisition_sampler: (
            str | AcquisitionSampler | type[AcquisitionSampler]
        ) = "random",
        patience: int = 100,
    ):
        surrogate_model_args = dict(surrogate_model_args) if surrogate_model_args else {}

        graph_kernels, hp_kernels = get_kernels(
            pipeline_space=space,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
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
            # TODO: Graph gets ignored?
            surrogate_model_args["vectorial_features"] = {
                "continuous": len(space.numericals),
                "categorical": len(space.categoricals),
            }

        # TODO: What the hell type is this
        self.surrogate_model: Any = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs=surrogate_model_args,
        )

        self.acquisition: BaseAcquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,  # type: ignore
            name="acquisition function",
        )

        self.acquisition_sampler: AcquisitionSampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,  # type: ignore
            name="acquisition sampler function",
            kwargs={"patience": patience, "pipeline_space": space},
        )
        self.uniform_sampler = UniformSampler.new(space)

    def _fantasize_pending(self, train_x, train_y, pending_x):
        if len(pending_x) == 0:
            return train_x, train_y

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
        self,
        n: int,
        *,
        active_max_fidelity: Mapping[str, Number] | None = None,
        fidelity: Mapping[str, Number] | None = None,
        seed: np.random.Generator,
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
        logger.info("Acquiring...")

        # sampling random configurations
        samples = [
            self.space.sample(user_priors=False, ignore_fidelity=True)
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
        # TODO: can generalize s.t. sampler works for all types, currently,
        #  random sampler in NePS does not do what is required here
        # return self.acquisition_sampler.sample(self.acquisition)
