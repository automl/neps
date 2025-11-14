from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.parego import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from gpytorch.utils.warnings import NumericalWarning

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.priorband import PriorBandSampler
from neps.optimizers.utils.brackets import PromoteAction, SampleAction
from neps.optimizers.utils.util import (
    get_config_key_to_id_mapping,
    get_trial_config_unique_key,
)
from neps.sampling.samplers import Sampler
from neps.utils.common import disable_warnings

if TYPE_CHECKING:
    from gpytorch.models.approximate_gp import Any

    from neps.optimizers.mopriors import MOPriorSampler
    from neps.optimizers.utils.brackets import Bracket
    from neps.space import SearchSpace
    from neps.space.encoding import ConfigEncoder
    from neps.space.parameters import Parameter
    from neps.state.optimizer import BudgetInfo
    from neps.state.pipeline_eval import UserResultDict
    from neps.state.seed_snapshot import RNGStateManager
    from neps.state.trial import Trial


logger = logging.getLogger(__name__)


def trials_to_table(trials: Mapping[str, Trial]) -> pd.DataFrame:
    id_index = np.empty(len(trials), dtype=int)
    rungs_index = np.empty(len(trials), dtype=int)
    perfs = [np.nan] * len(trials)
    configs = np.empty(len(trials), dtype=object)

    for i, (trial_id, trial) in enumerate(trials.items()):
        config_id_str, rung_str = trial_id.split("_rung_")
        _id, _rung = int(config_id_str), int(rung_str)

        if trial.report is None:
            perf = np.nan  # Pending
        elif trial.report.objective_to_minimize is None:
            perf = np.inf  # Error? Either way, we wont promote it
        elif isinstance(trial.report.objective_to_minimize, float):
            perf = trial.report.objective_to_minimize
        elif isinstance(trial.report.objective_to_minimize, Sequence):
            perf = np.array(trial.report.objective_to_minimize, dtype=np.float64)
        else:
            raise ValueError("Unknown type of objective_to_minimize")

        id_index[i] = _id
        rungs_index[i] = _rung
        perfs[i] = perf
        configs[i] = trial.config

    id_index = pd.MultiIndex.from_arrays([id_index, rungs_index], names=["id", "rung"])
    df = pd.DataFrame(data={"config": configs, "perf": perfs}, index=id_index)
    return df.sort_index(ascending=True)


@dataclass
class GPSampler:
    """See the following reference.

    PriorBand Appendix E.4, Model extensions,
    https://openreview.net/attachment?id=uoiwugtpCH&name=supplementary_material
    """

    parameters: Mapping[str, Parameter]
    """The parameters to use."""

    encoder: ConfigEncoder
    """The encoder to use for encoding and decoding configurations."""

    threshold: float
    """The threshold at which to switch to the Bayesian optimizer.

    This is calculated in the following way:
    * 1 `fid_unit` is equal to `fid_max`.
    * The minimum fidelity is equal to `fid_min / fid_max`.
    * BO Sampling kicks in after `threshold` units of `fit_unit` have been used.
    """

    two_stage_batch_sample_size: int
    """When fitting a GP jointly across all fidelitys, we do a two stage acquisition.

    For simplicity in writing, lets assume `two_stage_batch_sample_size` is 10.

    In the first stage, we acquire from the GP, the `10`
    configurations that are predicted to be best at that target fidelity.
    We then use another expected improvement on these
    `10` configurations, but with their fidelity set
    to the maximum, essentially to predict which one of those `10` configurations
    will be best at the maximum fidelity.
    """

    fidelity_name: str
    """The name of the fidelity in the space."""

    fidelity_max: int | float
    """The maximum fidelity value."""

    device: torch.device | None
    """The device to use for the GP optimization."""

    def threshold_reached(self, trials: Mapping[str, Trial]) -> bool:
        used_fidelity = [
            t.config[self.fidelity_name] for t in trials.values() if t.report is not None
        ]
        fidelity_units_used = sum(used_fidelity) / self.fidelity_max
        return fidelity_units_used >= self.threshold

    def sample_config(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        target_fidelity: int | float,
        seed: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Samples a configuration using the GP model.

        Please see parameter descriptions in the class docstring for more.
        """
        assert budget_info is None, "cost-aware (using budget_info) not supported yet."
        # fit the GP model using all trials, using fidelity as a dimension.
        # Get to top 10 configurations for acquisition fixed at fidelity Z
        # Switch those configurations to be at fidelity z_max and take the best.
        # y_max for EI is taken to be the best value seen so far, across all fidelity
        data, _ = encode_trials_for_gp(
            trials,
            self.parameters,
            encoder=self.encoder,
            device=self.device,
        )
        gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=self.encoder)

        with disable_warnings(NumericalWarning):
            acqf = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device)
                ),
                X_pending=data.x_pending,
                prune_baseline=True,
            )

        # When it's max fidelity, we can just sample the best configuration we find,
        # as we do not need to do the two step procedure.
        requires_two_step = target_fidelity != self.fidelity_max
        N = 1 if requires_two_step else self.two_stage_batch_sample_size

        candidates = fit_and_acquire_from_gp(
            gp=gp,
            encoder=self.encoder,
            x_train=data.x,
            n_candidates_required=N,
            acquisition=acqf,
            # Ensure we fix that acquisition happens at target fidelity
            fixed_acq_features={self.fidelity_name: target_fidelity},
            # NOTE: We don't support any cost aware or prior based GP stuff here
            # TODO: Theoretically, we could. Check out the implementation of
            # `BayesianOptimization` for more details
            prior=None,
            pibo_exp_term=None,
            costs=None,
            cost_percentage_used=None,
            costs_on_log_scale=False,
            hide_warnings=True,
            seed=seed,
        )
        assert len(candidates) == N

        # We bail out here, as we already acquired over max fidelity.
        if not requires_two_step:
            config = self.encoder.decode_one(candidates[0])
            assert config[self.fidelity_name] == target_fidelity, (
                f"Expected the target fidelity to be {target_fidelity}, "
                f"but got {config[self.fidelity_name]} for config: {config}"
            )
            return config

        # Next, we set those N configurations to be at the max fidelity
        # Decode, set max fidelity, and encode again (TODO: Could do directly on tensors)
        configs = self.encoder.decode(candidates)
        fid_max_configs = [{**c, self.fidelity_name: self.fidelity_max} for c in configs]
        encoded_fix_max_configs = self.encoder.encode(fid_max_configs)

        ys = acqf(encoded_fix_max_configs)
        idx_max = torch.argmax(ys)
        config = configs[idx_max]
        config.update({self.fidelity_name: target_fidelity})
        return config


@dataclass
class BracketOptimizer:
    """Implements an optimizer over brackets.

    This is the main class behind algorithms like `"priorband"`,
    `"successive_halving"`, `"asha"`, `"hyperband"`, etc.
    """

    space: SearchSpace
    """The pipeline space to optimize over."""

    encoder: ConfigEncoder
    """The encoder to use for the pipeline space."""

    sample_prior_first: bool | Literal["highest_fidelity"]
    """Whether or not to sample the prior first.

    If set to `"highest_fidelity"`, the prior will be sampled at the highest fidelity,
    otherwise at the lowest fidelity.
    """

    eta: int
    """The eta parameter for the algorithm."""

    rung_to_fid: Mapping[int, int | float]
    """The mapping from rung to fidelity value."""

    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]
    """A function that creates the brackets from the table of trials."""

    sampler: Sampler | PriorBandSampler | MOPriorSampler
    """The sampler used to generate new trials."""

    gp_sampler: GPSampler | None
    """If set, uses a GP for sampling configurations once it's threshold for
    fidelity units has been reached.
    """

    fid_min: int | float
    """The minimum fidelity value."""

    fid_max: int | float
    """The maximum fidelity value."""

    fid_name: str
    """The name of the fidelity in the space."""

    rng_manager: RNGStateManager
    """The RNG state manager to use for seeding."""

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "paramter n should be not None"
        space = self.space
        parameters = space.searchables
        # If we have no trials, we either go with the prior or just a sampled config
        if len(trials) == 0:
            match self.sample_prior_first:
                case "highest_fidelity":  # fid_max
                    config = {
                        name: p.prior if p.prior is not None else p.center
                        for name, p in parameters.items()
                    }
                    config.update(space.constants)
                    config[self.fid_name] = self.fid_max
                    rung = max(self.rung_to_fid)
                    return SampledConfig(id=f"1_rung_{rung}", config=config)
                case True:  # fid_min
                    config = {
                        name: p.prior if p.prior is not None else p.center
                        for name, p in parameters.items()
                    }
                    config.update(space.constants)
                    config[self.fid_name] = self.fid_min
                    rung = min(self.rung_to_fid)
                    return SampledConfig(id=f"1_rung_{rung}", config=config)
                case False:
                    pass

        table = trials_to_table(trials=trials)

        if len(table) == 0:  # noqa: SIM108
            # Nothing there, this sample will be the first
            nxt_id = 1
        else:
            # One plus the maximum current id in the table index
            nxt_id = table.index.get_level_values("id").max() + 1  # type: ignore

        # We don't want the first highest fidelity sample ending
        # up in a bracket
        if self.sample_prior_first == "highest_fidelity":
            table = table.iloc[1:]

        # Get and execute the next action from our brackets that are not pending or done
        assert isinstance(table, pd.DataFrame)
        brackets = self.create_brackets(table)

        if not isinstance(brackets, Sequence):
            brackets = [brackets]

        next_action = next(
            (
                action
                for bracket in brackets
                if (action := bracket.next()) not in ("done", "pending")
            ),
            None,
        )

        if next_action is None:
            raise RuntimeError(
                f"{self.__class__.__name__} never got a 'sample' or 'promote' action!"
                f" This likely means the implementation of {self.create_brackets}"
                " is incorrect and should have provided enough brackets, where at"
                " least one of them should have requested another sample."
                f"\nBrackets:\n{brackets}"
            )

        match next_action:
            # The bracket would like us to promote a configuration
            case PromoteAction(config=config, id=config_id, new_rung=new_rung):
                config = {
                    **config,
                    **space.constants,
                    self.fid_name: self.rung_to_fid[new_rung],
                }
                return SampledConfig(
                    id=f"{config_id}_rung_{new_rung}",
                    config=config,
                    previous_config_id=f"{config_id}_rung_{new_rung - 1}",
                )

            # The bracket would like us to sample a new configuration for a rung
            # and we have gp sampler which should kick in
            case SampleAction(rung=rung) if (
                self.gp_sampler is not None and self.gp_sampler.threshold_reached(trials)
            ):
                # If we should used BO to sample once a threshold has been reached,
                # do so. Otherwise we proceed to use the original sampler.
                target_fidelity = self.rung_to_fid[rung]
                config = self.gp_sampler.sample_config(
                    trials,
                    budget_info=None,  # TODO: budget_info not supported yet
                    target_fidelity=target_fidelity,
                    seed=self.rng_manager.torch_manual_rng,
                )
                config.update(space.constants)
                return SampledConfig(id=f"{nxt_id}_rung_{rung}", config=config)

            # We need to sample for a new rung, with either no gp or it has
            # not yet kicked in.
            case SampleAction(rung=rung):
                # Otherwise, we proceed with the original sampler
                match self.sampler:
                    case Sampler():
                        config = self.sampler.sample_config(
                            to=self.encoder, seed=self.rng_manager.torch_manual_rng
                        )
                        config = {
                            **config,
                            **space.constants,
                            self.fid_name: self.rung_to_fid[rung],
                        }
                        return SampledConfig(id=f"{nxt_id}_rung_{rung}", config=config)

                    case PriorBandSampler():
                        config = self.sampler.sample_config(table, rung=rung)
                        config = {
                            **config,
                            **space.constants,
                            self.fid_name: self.rung_to_fid[rung],
                        }
                        return SampledConfig(id=f"{nxt_id}_rung_{rung}", config=config)
                    case _:
                        raise RuntimeError(f"Unknown sampler: {self.sampler}")
            case _:
                raise RuntimeError(f"Unknown bracket action: {next_action}")

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        rung_to_fid = self.rung_to_fid

        # Use trials_to_table to get all used config IDs
        table = trials_to_table(trials)
        used_ids = set(table.index.get_level_values("id").tolist())

        imported_configs = []
        config_to_id = get_config_key_to_id_mapping(table=table, fid_name=self.fid_name)

        for config, result in external_evaluations:
            fid_value = config[self.fid_name]
            if fid_value not in rung_to_fid.values():
                logger.warning(
                    f"Fidelity value {fid_value} not in known rung fidelities "
                    f"{list(rung_to_fid.values())}. Skipping config: {config}"
                )
                continue
            # create a  unique key for the config without the fidelity
            config_key = get_trial_config_unique_key(
                config=config, fid_name=self.fid_name
            )
            # Assign id if not already assigned
            if config_key not in config_to_id:
                next_id = max(used_ids, default=0) + 1
                config_to_id[config_key] = next_id
                used_ids.add(next_id)
            else:
                existing_id = config_to_id[config_key]
                # check if the other config with same key has the same fidelity
                try:
                    existing_config = table.xs(existing_id, level="id")["config"].iloc[0]
                    if existing_config[self.fid_name] == config[self.fid_name]:
                        logger.warning(
                            f"Duplicate configuration with same fidelity found: {config}"
                        )
                    continue
                except KeyError:
                    pass

            config_id = config_to_id[config_key]

            # Find the rung corresponding to the fidelity value in config
            rung = next((r for r, f in rung_to_fid.items() if f == fid_value), None)
            trial_id = f"{config_id}_rung_{rung}"
            imported_configs.append(
                ImportedConfig(
                    id=trial_id,
                    config=copy.deepcopy(config),
                    result=copy.deepcopy(result),
                )
            )
        return imported_configs
