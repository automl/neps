from __future__ import annotations

from metahyper.api import instance_from_map

from ...search_spaces.search_space import SearchSpace
from .acquisition_functions import AcquisitionMapping
from .optimizer import BayesianOptimization


class CostCooling(BayesianOptimization):
    USES_COST_MODEL = True
    """Implements a basic cost-cooling as described in
    "Cost-aware Bayesian Optimization" (https://arxiv.org/abs/2003.10870) by Lee et al."""

    def __init__(
        self, base_acquisition="EI", cost_cooling_sampler="cost_cooler", **kwargs
    ):
        super().__init__(acquisition=base_acquisition, **kwargs)

        if self.budget is None:
            raise ValueError("CostCooling needs a maximum budget value")

        self.cost_cooling_acquisition = instance_from_map(
            AcquisitionMapping,
            cost_cooling_sampler,
            name="cost cooling acquisition function",
            kwargs={"base_acquisition": self.acquisition},
        )

    def sample_configuration_from_model(self) -> tuple[SearchSpace, None, None]:
        config = self.acquisition_sampler.sample(
            self.cost_cooling_acquisition, constraint=self.sampling_constraint
        )
        return config, None, None

    def _update_optimizer_training_state(self):
        super()._update_optimizer_training_state()
        self.cost_cooling_acquisition.set_state(
            self.surrogate_model,
            alpha=1 - (self.used_budget / self.budget),
            cost_model=self.cost_model,
            update_base_model=False,
        )
