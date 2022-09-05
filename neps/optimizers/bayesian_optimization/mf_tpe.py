from __future__ import annotations

import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from metahyper.api import ConfigResult, instance_from_map

from ...search_spaces import CategoricalParameter, FloatParameter, IntegerParameter
from ...search_spaces.search_space import SearchSpace
from .. import BaseOptimizer
from .acquisition_samplers import AcquisitionSamplerMapping
from .acquisition_samplers.base_acq_sampler import AcquisitionSampler
from .models import SurrogateModelMapping


class MultiFidelityPriorWeightedTreeParzenEstimator(BaseOptimizer):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        use_priors: bool = False,
        prior_num_evals: float = 1.0,
        good_fraction: float = 0.15,
        random_interleave_prob: float = 0.333,
        initial_design_size: int = 0,
        prior_as_samples: bool = True,
        pending_as_bad: bool = True,
        surrogate_model: str = "kde",
        acquisition_sampler: str | AcquisitionSampler = "mutation",
        prior_draws: int = 1000,
        cost_per_fidelity: list = None,
        surrogate_model_args: dict = None,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
    ):
        """[summary]

        Args:
            pipeline_space: Space in which to search
            prior_num_evals (float, optional): [description]. Defaults to 1.0.
            good_fraction (float, optional): [description]. Defaults to 0.333.
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            initial_design_size: Number of 'x' samples that are to be evaluated before
                selecting a sample using a strategy instead of randomly. If there is a
                user prior, we can rely on the model from the very first iteration.
            prior_as_samples: Whether to sample from the KDE and incorporate that way, or
            just have the distribution be an linear combination of the KDE and the prior.
            Should be True if the prior happens to be unnormalized.
            pending_as_bad: Whether to treat pending observations as bad, assigning them to
            the bad KDE to encourage diversity among samples queried in parallel
            prior_draws: The number of samples drawn from the prior if there is one. This
            # does not affect the strength of the prior, just how accurately it
            # is reconstructed by the KDE.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            loss_value_on_error: Setting this and cost_value_on_error to any float will
                supress any error during bayesian optimization and will use given loss
                value instead. default: None
            cost_value_on_error: Setting this and loss_value_on_error to any float will
                supress any error during bayesian optimization and will use given cost
                value instead. default: None
            logger: logger object, or None to use the neps logger
        """
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
        )
        self.pipeline_space = pipeline_space
        if self.pipeline_space.has_fidelity:
            self.min_fidelity = pipeline_space.fidelity.lower
            self.max_fidelity = pipeline_space.fidelity.upper
            self.cost_per_fidelity = (
                cost_per_fidelity
                or np.arange(self.min_fidelity, self.max_fidelity + 1) / self.max_fidelity
            )

        else:
            self.cost_per_fidelity = [1]

        self.prior_num_evals = prior_num_evals
        self.good_fraction = good_fraction
        self._random_interleave_prob = random_interleave_prob
        self._initial_design_size = initial_design_size
        self._pending_as_bad = pending_as_bad
        self.prior_draws = prior_draws

        # We assume that the information conveyed per fidelity (and the cost) is linear in the
        # fidelity levels if nothing else is specified
        if surrogate_model != "kde":
            raise NotImplementedError(
                "Only supports KDEs for now. Could (maybe?) support binary classification in the future."
            )
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )
        surrogate_model_args = surrogate_model_args or {}

        param_types, num_values, logged_params, is_fidelity = self._get_types()
        surrogate_model_args["param_types"] = param_types
        surrogate_model_args["num_values"] = num_values
        surrogate_model_args["is_fidelity"] = is_fidelity
        surrogate_model_args["logged_params"] = logged_params

        # TODO consider the logged versions of parameters
        if self.pipeline_space.has_prior and use_priors:
            if prior_as_samples:
                self.prior_samples = [
                    self.pipeline_space.sample(
                        patience=self.patience, user_priors=True, ignore_fidelity=False
                    )
                    for idx in range(self.prior_draws)
                ]
            else:
                pass
                # TODO work out affine combination
        else:
            self.prior_samples = []

        self.surrogate_models = {
            "good": instance_from_map(
                SurrogateModelMapping,
                surrogate_model,
                name="surrogate model",
                kwargs=surrogate_model_args,
            ),
            "bad": instance_from_map(
                SurrogateModelMapping,
                surrogate_model,
                name="surrogate model",
                kwargs=surrogate_model_args,
            ),
            "all": instance_from_map(
                SurrogateModelMapping,
                surrogate_model,
                name="surrogate model",
                kwargs=surrogate_model_args,
            ),
        }
        self.acquisition = self
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )

    def _get_types(self):
        """extracts the needed types from the configspace for faster retrival later

        type = 0 - numerical (continuous or integer) parameter
        type >=1 - categorical parameter

        TODO: figure out a way to properly handle ordinal parameters

        """
        types = []
        num_values = []
        logs = []
        is_fidelity = []
        for _, hp in self.pipeline_space.items():
            is_fidelity.append(hp.is_fidelity)
            if isinstance(hp, CategoricalParameter):
                # u as in unordered - used to play nice with the statsmodels KDE implementation
                types.append("u")
                logs.append(False)
                num_values.append(len(hp.choices))
            elif isinstance(hp, IntegerParameter):
                # o as in ordered
                types.append("o")
                logs.append(False)
                num_values.append(hp.upper - hp.lower + 1)
            elif isinstance(hp, FloatParameter):
                # c as in continous
                types.append("c")
                logs.append(hp.log)
                num_values.append(np.inf)

            else:
                raise ValueError("Unsupported Parametertype %s" % type(hp))

        return types, num_values, logs, is_fidelity

    def __call__(
        self, x: Iterable, asscalar: bool = False
    ) -> np.ndarray | torch.Tensor | float:
        """
        Return the negative probability of / expected improvement at the query point
        """
        return self.surrogate_models["good"].pdf(x) / self.surrogate_models["bad"].pdf(x)

    # TODO allow this for integers as well - now only supports floats
    def _convert_to_logscale(self):
        pass

    def _split_by_fidelity(self, configs, losses):
        min_fid, max_fid = int(self.min_fidelity), int(self.max_fidelity)            
        if self.pipeline_space.has_fidelity:

            configs_per_fidelity = [[] for i in range(min_fid, max_fid + 1)]
            losses_per_fidelity = [[] for i in range(min_fid, max_fid + 1)]
            # per fidelity, add a list to make it a nested list of lists
            # [[config_A at fid1, config_B at fid1], [config_C at fid2], ...]
            for config, loss in zip(configs, losses):
                configs_per_fidelity[config.fidelity.value - min_fid].append(config)
                losses_per_fidelity[config.fidelity.value - min_fid].append(loss)
            return configs_per_fidelity, losses_per_fidelity
        else:
            return [configs], [losses]

    def _split_configs(self, configs, losses, round_up=True):
        """Splits configs into good and bad for the KDEs.

        Args:
            configs ([type]): [description]
            losses ([type]): [description]
            round_up (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # TODO split by fidelity
        print([c.fidelity.value for c in configs])
        good_configs, bad_configs = [], []
        good_configs_weights, bad_configs_weights = [], []
        configs_per_fid, losses_per_fid = self._split_by_fidelity(configs, losses)

        for fid, (configs_fid, losses_fid) in enumerate(
            zip(configs_per_fid, losses_per_fid)
        ):
            # TODO map to cost if available
            num_good_configs = np.ceil(len(configs_fid) * self.good_fraction).astype(int)
            if not round_up:
                num_good_configs -= 1

            ordered_losses = np.argsort(losses_fid)
            good_indices = ordered_losses[0:num_good_configs]
            bad_indices = ordered_losses[num_good_configs:]
            good_configs_fid = [configs_fid[idx] for idx in good_indices]
            bad_configs_fid = [configs_fid[idx] for idx in bad_indices]
            good_configs.extend(good_configs_fid)
            bad_configs.extend(bad_configs_fid)
            good_configs_weights.extend(
                [self.cost_per_fidelity[fid]] * len(good_configs_fid)
            )
            bad_configs_weights.extend(
                [self.cost_per_fidelity[fid]] * len(bad_configs_fid)
            )
            print('good_configs', good_configs)
            print('bad_configs', bad_configs)
            print('good_configs_weights', good_configs_weights)
            print('bad_configs_weights', bad_configs_weights)
        return good_configs, bad_configs, good_configs_weights, bad_configs_weights

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search."""
        if self._num_train_x >= self._initial_design_size:
            return False
        return True

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        train_x_configs = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]

        train_x_configs = [el.config for el in previous_results.values()]
        pending_x_configs = [el.config for el in pending_evaluations.values()]
        self._num_train_x = len(train_x_configs)
        self._pending_evaluations = pending_x_configs
        if not self.is_init_phase():
            # This is to extract the configurations as numpy arrays on the format num_data x num_dim
            good_configs, bad_configs, good_weights, bad_weights = self._split_configs(
                train_x_configs, train_y, round_up=True
            )

            num_good_configs = len(good_configs)
            num_prior_configs = len(self.prior_samples)
            good_configs.extend(self.prior_samples)
            good_config_weights = np.ones(len(good_configs))
            good_config_weights[num_good_configs:] = (
                self.prior_num_evals / num_prior_configs
            )
            # TODO drop the fidelity!
            self.surrogate_models["all"].fit(train_x_configs)
            fixed_bw = self.surrogate_models["all"].bw
            self.surrogate_models["good"].fit(
                good_configs, fixed_bw=fixed_bw, config_weights=good_config_weights
            )  # [num_good_configs:])
            if self._pending_as_bad:
                bad_configs.extend(pending_x_configs)
            self.surrogate_models["bad"].fit(bad_configs, fixed_bw=fixed_bw)

            self.visualize_acq(previous_results)

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        if self._num_train_x == 0 and self._initial_design_size >= 1:
            # TODO only at lowest fidelity
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        elif random.random() < self._random_interleave_prob:
            # TODO only at lowest fidelity
            config = self.pipeline_space.sample(
                patience=self.patience, ignore_fidelity=False
            )
        elif self.is_init_phase():
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        else:
            config = self.acquisition_sampler.sample(self.acquisition)
            if config not in self._pending_evaluations:
                # TODO fix this, so that it works properly
                pass

        config_id = str(self._num_train_x + len(self._pending_evaluations) + 1)
        return config.hp_values(), config_id, None

    def visualize_2d(
        self, ax, previous_results, grid_points: int = 101, color: str = "k"
    ):
        X1 = np.linspace(0, 1, grid_points)
        X2 = np.linspace(0, 1, grid_points)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.append(X1.reshape(-1, 1), X2.reshape(-1, 1), axis=1)
        Z = self.surrogate_models["good"]._pdf(X) / self.surrogate_models["bad"]._pdf(X)
        Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()

        Z = Z.reshape(grid_points, grid_points)

        c = ax.pcolormesh(X1, X2, Z, cmap=color, vmin=Z_min, vmax=Z_max)
        ax.set_title("pcolormesh")
        # set the limits of the plot to the limits of the data
        ax.axis([0, 1, 0, 1])
        train_x_configs = [el.config for el in previous_results.values()]
        np_X = self.surrogate_models["good"]._convert_configs_to_numpy(train_x_configs)
        ax.scatter(np_X[:, 0], np_X[:, 1], s=100)
        # ax.scatter(np_X[-1, 0], np_X[-1, 1], s=100, c='yellow')

        return ax

    def visualize_acq(self, previous_results):
        train_x_configs = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]

        (
            good_configs,
            bad_configs,
            good_configs_weights,
            bad_configs_weights,
        ) = self._split_configs(train_x_configs, train_y, round_up=True)
        good_configs_np = self.surrogate_models["all"]._convert_configs_to_numpy(
            good_configs
        )
        bad_configs_np = self.surrogate_models["all"]._convert_configs_to_numpy(
            bad_configs
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 9))
        axes[0] = self.surrogate_models["good"].visualize_2d(axes[0], color="RdBu")
        axes[0].scatter(
            good_configs_np[:, 0], good_configs_np[:, 1], c="orange", s=50, marker="x"
        )
        axes[1] = self.surrogate_models["bad"].visualize_2d(axes[1], color="RdBu_r")
        axes[1].scatter(
            bad_configs_np[:, 0], bad_configs_np[:, 1], c="orange", s=50, marker="x"
        )
        axes[2] = self.visualize_2d(axes[2], previous_results, color="BrBG")
        plt.show()
