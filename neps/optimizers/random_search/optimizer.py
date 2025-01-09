"""Random search optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.sampling.priors import UniformPrior
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


class RandomSearch(BaseOptimizer):
    """A simple random search optimizer."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        use_priors: bool = False,
        ignore_fidelity: bool = True,
        seed: int | None = None,
        **kwargs: Any,  # TODO: Remove
    ):
        """Initialize the random search optimizer.

        Args:
            pipeline_space: The search space to sample from.
            use_priors: Whether to use priors when sampling.
            ignore_fidelity: Whether to ignore fidelity when sampling.
                In this case, the max fidelity is always used.
            seed: The seed for the random number generator.
        """
        super().__init__(pipeline_space=pipeline_space)
        self.use_priors = use_priors
        self.ignore_fidelity = ignore_fidelity
        if seed is not None:
            raise NotImplementedError("Seed is not implemented yet for RandomSearch")

        self.seed = seed
        self.encoder = ConfigEncoder.from_space(
            pipeline_space,
            include_fidelity=False,
            include_constants_when_decoding=True,
        )
        self.sampler = UniformPrior(ndim=self.encoder.ncols)

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        n_trials = len(trials)
        _n = 1 if n is None else n
        configs = self.sampler.sample(_n, to=self.encoder.domains)
        config_dicts = self.encoder.decode(configs)
        if n == 1:
            config = config_dicts[0]
            config_id = str(n_trials + 1)
            return SampledConfig(config=config, id=config_id, previous_config_id=None)

        return [
            SampledConfig(
                config=config,
                id=str(n_trials + i + 1),
                previous_config_id=None,
            )
            for i, config in enumerate(config_dicts)
        ]
